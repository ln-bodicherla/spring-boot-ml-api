package com.portfolio.mlapi.service;

import ai.onnxruntime.*;
import com.portfolio.mlapi.config.ModelConfig;
import com.portfolio.mlapi.model.PredictionRequest;
import io.github.resilience4j.circuitbreaker.annotation.CircuitBreaker;
import io.github.resilience4j.retry.annotation.Retry;
import io.micrometer.core.instrument.Counter;
import io.micrometer.core.instrument.MeterRegistry;
import io.micrometer.core.instrument.Timer;
import jakarta.annotation.PostConstruct;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.nio.FloatBuffer;
import java.time.Instant;
import java.util.*;
import java.util.concurrent.atomic.AtomicLong;
import java.util.stream.Collectors;

/**
 * Core model inference service with circuit breaker fault tolerance.
 *
 * Manages the ONNX Runtime session lifecycle, performs model inference,
 * and exposes Prometheus metrics for prediction latency and throughput.
 */
@Service
public class ModelService {

    private static final Logger logger = LoggerFactory.getLogger(ModelService.class);

    private static final double FRAUD_THRESHOLD = 0.5;
    private static final String LABEL_FRAUD = "fraud";
    private static final String LABEL_LEGITIMATE = "legitimate";

    private final OrtEnvironment ortEnvironment;
    private final OrtSession ortSession;
    private final PreprocessingService preprocessingService;
    private final ModelConfig modelConfig;

    private final Timer inferenceTimer;
    private final Counter predictionCounter;
    private final Counter errorCounter;
    private final AtomicLong totalPredictions = new AtomicLong(0);
    private final AtomicLong totalInferenceTimeMs = new AtomicLong(0);

    private Instant loadedAt;
    private boolean modelLoaded;

    @Value("${model.warmup-samples:10}")
    private int warmupSamples;

    @Autowired
    public ModelService(
            OrtEnvironment ortEnvironment,
            OrtSession ortSession,
            PreprocessingService preprocessingService,
            ModelConfig modelConfig,
            MeterRegistry meterRegistry
    ) {
        this.ortEnvironment = ortEnvironment;
        this.ortSession = ortSession;
        this.preprocessingService = preprocessingService;
        this.modelConfig = modelConfig;

        this.inferenceTimer = Timer.builder("prediction.latency")
                .description("Model inference latency")
                .tag("model", modelConfig.getModelName())
                .register(meterRegistry);

        this.predictionCounter = Counter.builder("prediction.requests")
                .description("Total prediction requests")
                .tag("model", modelConfig.getModelName())
                .register(meterRegistry);

        this.errorCounter = Counter.builder("prediction.errors")
                .description("Total prediction errors")
                .tag("model", modelConfig.getModelName())
                .register(meterRegistry);

        this.modelLoaded = (ortSession != null);
        this.loadedAt = Instant.now();
    }

    /**
     * Warm up the model on startup with synthetic predictions.
     *
     * Ensures the ONNX Runtime session and thread pools are
     * initialized before serving real traffic to minimize
     * cold-start latency.
     */
    @PostConstruct
    public void warmUp() {
        if (!modelLoaded) {
            logger.info("Model not loaded, skipping warmup. Running in demo mode.");
            return;
        }

        logger.info("Warming up model with {} samples...", warmupSamples);
        Random random = new Random(42);

        int successCount = 0;
        for (int i = 0; i < warmupSamples; i++) {
            try {
                List<Double> features = new ArrayList<>();
                for (int j = 0; j < modelConfig.getInputSize(); j++) {
                    features.add(random.nextGaussian());
                }
                runInference(features, null);
                successCount++;
            } catch (Exception e) {
                logger.warn("Warmup sample {} failed: {}", i, e.getMessage());
            }
        }

        totalPredictions.set(0);
        totalInferenceTimeMs.set(0);

        logger.info("Model warmup complete: {}/{} samples succeeded", successCount, warmupSamples);
    }

    /**
     * Run a single prediction with circuit breaker protection.
     *
     * Preprocesses input features, runs ONNX inference, and
     * returns a structured prediction response. Falls back to
     * an error response if the circuit breaker is open.
     *
     * @param request the prediction request with feature values
     * @return prediction response with score, label, and metadata
     */
    @CircuitBreaker(name = "modelService", fallbackMethod = "predictFallback")
    @Retry(name = "modelService")
    public PredictionRequest.PredictionResponse predict(PredictionRequest.SingleRequest request) {
        long startTime = System.currentTimeMillis();
        predictionCounter.increment();

        String requestId = request.getRequestId() != null
                ? request.getRequestId()
                : UUID.randomUUID().toString();

        List<String> validationErrors = preprocessingService.validateInput(request.getFeatures());
        if (!validationErrors.isEmpty()) {
            throw new IllegalArgumentException(
                    "Input validation failed: " + String.join("; ", validationErrors)
            );
        }

        float[] result = runInference(request.getFeatures(), request.getFeatureNames());

        double score = result[0];
        String label = score >= FRAUD_THRESHOLD ? LABEL_FRAUD : LABEL_LEGITIMATE;
        double confidence = score >= FRAUD_THRESHOLD ? score : 1.0 - score;

        long inferenceTimeMs = System.currentTimeMillis() - startTime;
        totalPredictions.incrementAndGet();
        totalInferenceTimeMs.addAndGet(inferenceTimeMs);
        inferenceTimer.record(java.time.Duration.ofMillis(inferenceTimeMs));

        return PredictionRequest.PredictionResponse.of(
                requestId,
                score,
                label,
                confidence,
                inferenceTimeMs,
                modelConfig.getModelVersion()
        );
    }

    /**
     * Run batch predictions with circuit breaker protection.
     *
     * Processes multiple prediction requests sequentially and
     * aggregates results into a batch response.
     *
     * @param batchRequest the batch request containing multiple feature vectors
     * @return batch response with all predictions and aggregate metrics
     */
    @CircuitBreaker(name = "modelService", fallbackMethod = "predictBatchFallback")
    public PredictionRequest.BatchPredictionResponse predictBatch(
            PredictionRequest.BatchRequest batchRequest
    ) {
        long batchStartTime = System.currentTimeMillis();

        List<PredictionRequest.PredictionResponse> predictions = batchRequest.getRequests()
                .stream()
                .map(request -> {
                    try {
                        return predict(request);
                    } catch (Exception e) {
                        errorCounter.increment();
                        logger.error("Batch prediction failed for request: {}", e.getMessage());
                        return PredictionRequest.PredictionResponse.of(
                                request.getRequestId(),
                                0.0,
                                "error",
                                0.0,
                                0,
                                modelConfig.getModelVersion()
                        );
                    }
                })
                .collect(Collectors.toList());

        long totalTime = System.currentTimeMillis() - batchStartTime;

        return PredictionRequest.BatchPredictionResponse.of(predictions, totalTime);
    }

    /**
     * Get model metadata and operational statistics.
     *
     * @return model info including version, input/output specs, and performance stats
     */
    public PredictionRequest.ModelInfoResponse getModelInfo() {
        List<String> inputNames = new ArrayList<>();
        List<String> outputNames = new ArrayList<>();
        Map<String, String> metadata = new LinkedHashMap<>();

        if (modelLoaded && ortSession != null) {
            try {
                inputNames.addAll(ortSession.getInputNames());
                outputNames.addAll(ortSession.getOutputNames());
                metadata.put("onnx_runtime_version", OrtEnvironment.getApiVersion());
            } catch (Exception e) {
                logger.warn("Could not read session metadata: {}", e.getMessage());
            }
        }

        metadata.put("status", modelLoaded ? "loaded" : "demo_mode");
        metadata.put("threads", String.valueOf(modelConfig.getNumThreads()));

        long totalPreds = totalPredictions.get();
        double avgTime = totalPreds > 0
                ? (double) totalInferenceTimeMs.get() / totalPreds
                : 0.0;

        return PredictionRequest.ModelInfoResponse.builder()
                .modelName(modelConfig.getModelName())
                .modelVersion(modelConfig.getModelVersion())
                .modelPath(modelConfig.getModelPath())
                .inputSize(modelConfig.getInputSize())
                .inputNames(inputNames)
                .outputNames(outputNames)
                .metadata(metadata)
                .loaded(modelLoaded)
                .loadedAt(loadedAt)
                .totalPredictions(totalPreds)
                .avgInferenceTimeMs(Math.round(avgTime * 100.0) / 100.0)
                .build();
    }

    /**
     * Run the actual ONNX inference session.
     *
     * Preprocesses features, creates the ONNX tensor, runs inference,
     * and extracts the prediction score. Uses mock inference in demo mode.
     */
    private float[] runInference(List<Double> features, List<String> featureNames) {
        float[] normalized = preprocessingService.normalizeFeatures(features, featureNames);

        if (!modelLoaded || ortSession == null) {
            return mockInference(normalized);
        }

        try {
            long[] shape = new long[]{1, normalized.length};
            OnnxTensor inputTensor = OnnxTensor.createTensor(
                    ortEnvironment,
                    FloatBuffer.wrap(normalized),
                    shape
            );

            String inputName = ortSession.getInputNames().iterator().next();

            try (OrtSession.Result result = ortSession.run(
                    Collections.singletonMap(inputName, inputTensor)
            )) {
                float[][] output = (float[][]) result.get(0).getValue();
                inputTensor.close();

                if (output[0].length >= 2) {
                    float expFraud = (float) Math.exp(output[0][1]);
                    float expLegit = (float) Math.exp(output[0][0]);
                    return new float[]{expFraud / (expFraud + expLegit)};
                } else {
                    float score = 1.0f / (1.0f + (float) Math.exp(-output[0][0]));
                    return new float[]{score};
                }
            }
        } catch (OrtException e) {
            errorCounter.increment();
            logger.error("ONNX inference failed: {}", e.getMessage());
            throw new RuntimeException("Model inference failed: " + e.getMessage(), e);
        }
    }

    /**
     * Mock inference for demo mode when no ONNX model is loaded.
     *
     * Computes a deterministic score based on input features
     * using a simple linear combination with sigmoid activation.
     */
    private float[] mockInference(float[] features) {
        double weightedSum = 0.0;
        double[] weights = {0.3, -0.1, 0.5, -0.2, 0.4, -0.3, 0.2, 0.1, -0.4, 0.15};

        for (int i = 0; i < features.length; i++) {
            double weight = i < weights.length ? weights[i] : 0.1;
            weightedSum += features[i] * weight;
        }

        float score = (float) (1.0 / (1.0 + Math.exp(-weightedSum)));
        return new float[]{score};
    }

    /**
     * Circuit breaker fallback for single predictions.
     */
    private PredictionRequest.PredictionResponse predictFallback(
            PredictionRequest.SingleRequest request,
            Throwable throwable
    ) {
        logger.warn("Circuit breaker activated for predict: {}", throwable.getMessage());
        errorCounter.increment();

        return PredictionRequest.PredictionResponse.of(
                request.getRequestId() != null ? request.getRequestId() : "unknown",
                -1.0,
                "unavailable",
                0.0,
                0,
                modelConfig.getModelVersion()
        );
    }

    /**
     * Circuit breaker fallback for batch predictions.
     */
    private PredictionRequest.BatchPredictionResponse predictBatchFallback(
            PredictionRequest.BatchRequest batchRequest,
            Throwable throwable
    ) {
        logger.warn("Circuit breaker activated for batch predict: {}", throwable.getMessage());
        errorCounter.increment();

        List<PredictionRequest.PredictionResponse> fallbackResponses = batchRequest.getRequests()
                .stream()
                .map(req -> PredictionRequest.PredictionResponse.of(
                        req.getRequestId(), -1.0, "unavailable", 0.0, 0, modelConfig.getModelVersion()
                ))
                .collect(Collectors.toList());

        return PredictionRequest.BatchPredictionResponse.of(fallbackResponses, 0);
    }

    /**
     * Check if the model is loaded and ready for inference.
     */
    public boolean isModelLoaded() {
        return modelLoaded;
    }
}
