package com.portfolio.mlapi.config;

import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import jakarta.annotation.PreDestroy;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Spring configuration for ONNX Runtime model loading.
 *
 * Manages the ONNX Runtime environment and session lifecycle,
 * configuring thread pools and execution providers for optimal
 * inference performance.
 */
@Configuration
public class ModelConfig {

    private static final Logger logger = LoggerFactory.getLogger(ModelConfig.class);

    @Value("${model.path:models/fraud_model.onnx}")
    private String modelPath;

    @Value("${model.num-threads:4}")
    private int numThreads;

    @Value("${model.name:fraud-detection}")
    private String modelName;

    @Value("${model.version:1.0.0}")
    private String modelVersion;

    @Value("${model.input-size:10}")
    private int inputSize;

    private OrtEnvironment environment;
    private OrtSession session;

    /**
     * Create the ONNX Runtime environment singleton.
     *
     * The environment manages global thread pools and logging
     * for all ONNX Runtime sessions.
     */
    @Bean
    public OrtEnvironment ortEnvironment() {
        logger.info("Initializing ONNX Runtime environment");
        environment = OrtEnvironment.getEnvironment();
        logger.info("ONNX Runtime environment initialized");
        return environment;
    }

    /**
     * Create and configure the ONNX Runtime inference session.
     *
     * Loads the model file, configures inter/intra-op thread counts,
     * and enables graph optimizations for production inference.
     * Falls back to a placeholder session if the model file is not found.
     */
    @Bean
    public OrtSession ortSession(OrtEnvironment env) throws OrtException {
        Path path = Path.of(modelPath);

        OrtSession.SessionOptions options = new OrtSession.SessionOptions();
        options.setIntraOpNumThreads(numThreads);
        options.setInterOpNumThreads(Math.max(1, numThreads / 2));
        options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
        options.addConfigEntry("session.intra_op.allow_spinning", "1");

        if (Files.exists(path)) {
            logger.info("Loading ONNX model from: {} (threads: {})", modelPath, numThreads);
            session = env.createSession(modelPath, options);

            logger.info("Model loaded - Inputs: {}, Outputs: {}",
                    session.getInputNames(),
                    session.getOutputNames());
        } else {
            logger.warn("Model file not found: {}. Running in demo mode with mock inference.", modelPath);
            session = null;
        }

        return session;
    }

    /**
     * Clean up ONNX Runtime resources on shutdown.
     */
    @PreDestroy
    public void cleanup() {
        logger.info("Shutting down ONNX Runtime resources");
        if (session != null) {
            try {
                session.close();
            } catch (OrtException e) {
                logger.error("Error closing ONNX session", e);
            }
        }
        if (environment != null) {
            environment.close();
        }
        logger.info("ONNX Runtime resources released");
    }

    public String getModelPath() {
        return modelPath;
    }

    public String getModelName() {
        return modelName;
    }

    public String getModelVersion() {
        return modelVersion;
    }

    public int getInputSize() {
        return inputSize;
    }

    public int getNumThreads() {
        return numThreads;
    }
}
