package com.portfolio.mlapi.controller;

import com.portfolio.mlapi.model.PredictionRequest;
import com.portfolio.mlapi.service.ModelService;
import io.micrometer.core.annotation.Timed;
import jakarta.validation.Valid;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.validation.FieldError;
import org.springframework.web.bind.MethodArgumentNotValidException;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.context.request.WebRequest;

import java.lang.management.ManagementFactory;
import java.time.Instant;
import java.util.HashMap;
import java.util.Map;

/**
 * REST controller for ML model prediction endpoints.
 *
 * Provides single and batch prediction endpoints, model info retrieval,
 * and health checks. Includes global exception handling for input
 * validation errors and inference failures.
 */
@RestController
@RequestMapping("/api/v1")
public class PredictionController {

    private static final Logger logger = LoggerFactory.getLogger(PredictionController.class);

    private final ModelService modelService;

    @Autowired
    public PredictionController(ModelService modelService) {
        this.modelService = modelService;
    }

    /**
     * Single prediction endpoint.
     *
     * Accepts a feature vector and returns a fraud prediction with
     * score, label, confidence, and inference timing metadata.
     *
     * @param request prediction request with feature values
     * @return prediction response with fraud score and label
     */
    @PostMapping("/predict")
    @Timed(value = "api.predict", description = "Single prediction endpoint latency")
    public ResponseEntity<PredictionRequest.PredictionResponse> predict(
            @Valid @RequestBody PredictionRequest.SingleRequest request
    ) {
        logger.info(
                "Prediction request received: {} features, requestId={}",
                request.getFeatures().size(),
                request.getRequestId()
        );

        PredictionRequest.PredictionResponse response = modelService.predict(request);

        logger.info(
                "Prediction completed: requestId={}, label={}, score={:.4f}, time={}ms",
                response.getRequestId(),
                response.getLabel(),
                response.getScore(),
                response.getInferenceTimeMs()
        );

        return ResponseEntity.ok(response);
    }

    /**
     * Batch prediction endpoint.
     *
     * Accepts multiple feature vectors and returns predictions for
     * each. Includes aggregate metrics (success/failure counts,
     * total inference time).
     *
     * @param batchRequest batch request containing multiple prediction requests
     * @return batch response with all predictions and summary statistics
     */
    @PostMapping("/predict/batch")
    @Timed(value = "api.predict.batch", description = "Batch prediction endpoint latency")
    public ResponseEntity<PredictionRequest.BatchPredictionResponse> predictBatch(
            @Valid @RequestBody PredictionRequest.BatchRequest batchRequest
    ) {
        logger.info(
                "Batch prediction request received: {} requests",
                batchRequest.getRequests().size()
        );

        PredictionRequest.BatchPredictionResponse response = modelService.predictBatch(batchRequest);

        logger.info(
                "Batch prediction completed: {}/{} succeeded, total time={}ms",
                response.getSuccessCount(),
                response.getTotalRequests(),
                response.getTotalInferenceTimeMs()
        );

        return ResponseEntity.ok(response);
    }

    /**
     * Model information endpoint.
     *
     * Returns metadata about the loaded model including version,
     * input/output specifications, and operational statistics.
     *
     * @return model info response
     */
    @GetMapping("/model/info")
    @Timed(value = "api.model.info", description = "Model info endpoint latency")
    public ResponseEntity<PredictionRequest.ModelInfoResponse> getModelInfo() {
        PredictionRequest.ModelInfoResponse info = modelService.getModelInfo();
        return ResponseEntity.ok(info);
    }

    /**
     * Application health check endpoint.
     *
     * Returns service health status including model load state,
     * circuit breaker status, and uptime metrics.
     *
     * @return health response
     */
    @GetMapping("/health")
    public ResponseEntity<PredictionRequest.HealthResponse> healthCheck() {
        long uptimeMs = ManagementFactory.getRuntimeMXBean().getUptime();
        long uptimeSeconds = uptimeMs / 1000;

        String modelStatus = modelService.isModelLoaded() ? "loaded" : "demo_mode";
        String circuitBreakerStatus = "closed";

        PredictionRequest.HealthResponse response = PredictionRequest.HealthResponse.builder()
                .status("UP")
                .modelStatus(modelStatus)
                .circuitBreakerStatus(circuitBreakerStatus)
                .timestamp(Instant.now())
                .uptimeSeconds(uptimeSeconds)
                .build();

        return ResponseEntity.ok(response);
    }

    /**
     * Global exception handler for validation errors.
     *
     * Converts Spring validation errors into structured error
     * responses with field-level detail.
     */
    @ExceptionHandler(MethodArgumentNotValidException.class)
    @ResponseStatus(HttpStatus.BAD_REQUEST)
    public PredictionRequest.ErrorResponse handleValidationExceptions(
            MethodArgumentNotValidException ex,
            WebRequest request
    ) {
        Map<String, String> fieldErrors = new HashMap<>();
        ex.getBindingResult().getAllErrors().forEach(error -> {
            String fieldName = ((FieldError) error).getField();
            String errorMessage = error.getDefaultMessage();
            fieldErrors.put(fieldName, errorMessage);
        });

        String message = "Validation failed: " + fieldErrors;
        logger.warn("Validation error: {}", message);

        return PredictionRequest.ErrorResponse.of(
                HttpStatus.BAD_REQUEST.value(),
                "Bad Request",
                message,
                request.getDescription(false)
        );
    }

    /**
     * Global exception handler for illegal argument errors.
     *
     * Catches input validation failures from the preprocessing
     * service and returns structured error responses.
     */
    @ExceptionHandler(IllegalArgumentException.class)
    @ResponseStatus(HttpStatus.BAD_REQUEST)
    public PredictionRequest.ErrorResponse handleIllegalArgumentException(
            IllegalArgumentException ex,
            WebRequest request
    ) {
        logger.warn("Invalid input: {}", ex.getMessage());

        return PredictionRequest.ErrorResponse.of(
                HttpStatus.BAD_REQUEST.value(),
                "Bad Request",
                ex.getMessage(),
                request.getDescription(false)
        );
    }

    /**
     * Global exception handler for runtime errors during inference.
     *
     * Catches model inference failures and returns appropriate
     * error responses without exposing internal details.
     */
    @ExceptionHandler(RuntimeException.class)
    @ResponseStatus(HttpStatus.INTERNAL_SERVER_ERROR)
    public PredictionRequest.ErrorResponse handleRuntimeException(
            RuntimeException ex,
            WebRequest request
    ) {
        logger.error("Inference error: {}", ex.getMessage(), ex);

        return PredictionRequest.ErrorResponse.of(
                HttpStatus.INTERNAL_SERVER_ERROR.value(),
                "Internal Server Error",
                "Model inference failed. Please try again later.",
                request.getDescription(false)
        );
    }

    /**
     * Catch-all exception handler for unexpected errors.
     */
    @ExceptionHandler(Exception.class)
    @ResponseStatus(HttpStatus.INTERNAL_SERVER_ERROR)
    public PredictionRequest.ErrorResponse handleGenericException(
            Exception ex,
            WebRequest request
    ) {
        logger.error("Unexpected error: {}", ex.getMessage(), ex);

        return PredictionRequest.ErrorResponse.of(
                HttpStatus.INTERNAL_SERVER_ERROR.value(),
                "Internal Server Error",
                "An unexpected error occurred.",
                request.getDescription(false)
        );
    }
}
