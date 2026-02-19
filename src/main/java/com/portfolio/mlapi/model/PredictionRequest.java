package com.portfolio.mlapi.model;

import jakarta.validation.Valid;
import jakarta.validation.constraints.NotEmpty;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Size;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.Instant;
import java.util.List;
import java.util.Map;

/**
 * Request and response DTOs for the ML prediction API.
 *
 * Uses Jakarta validation annotations for input validation and
 * Lombok for boilerplate reduction.
 */
public class PredictionRequest {

    /**
     * Single prediction request containing feature values.
     */
    @Data
    @Builder
    @NoArgsConstructor
    @AllArgsConstructor
    public static class SingleRequest {

        @NotNull(message = "Features array must not be null")
        @NotEmpty(message = "Features array must not be empty")
        @Size(min = 1, max = 1000, message = "Features must contain between 1 and 1000 values")
        private List<Double> features;

        private List<String> featureNames;

        private String requestId;
    }

    /**
     * Batch prediction request containing multiple feature vectors.
     */
    @Data
    @Builder
    @NoArgsConstructor
    @AllArgsConstructor
    public static class BatchRequest {

        @NotNull(message = "Requests list must not be null")
        @NotEmpty(message = "Requests list must not be empty")
        @Size(max = 100, message = "Batch size must not exceed 100")
        @Valid
        private List<SingleRequest> requests;
    }

    /**
     * Single prediction response with score, label, and metadata.
     */
    @Data
    @Builder
    @NoArgsConstructor
    @AllArgsConstructor
    public static class PredictionResponse {

        private String requestId;
        private double score;
        private String label;
        private double confidence;
        private long inferenceTimeMs;
        private String modelVersion;
        private Instant timestamp;

        /**
         * Create a response for a successful prediction.
         */
        public static PredictionResponse of(
                String requestId,
                double score,
                String label,
                double confidence,
                long inferenceTimeMs,
                String modelVersion
        ) {
            return PredictionResponse.builder()
                    .requestId(requestId)
                    .score(score)
                    .label(label)
                    .confidence(confidence)
                    .inferenceTimeMs(inferenceTimeMs)
                    .modelVersion(modelVersion)
                    .timestamp(Instant.now())
                    .build();
        }
    }

    /**
     * Batch prediction response wrapping multiple individual results.
     */
    @Data
    @Builder
    @NoArgsConstructor
    @AllArgsConstructor
    public static class BatchPredictionResponse {

        private List<PredictionResponse> predictions;
        private int totalRequests;
        private int successCount;
        private int failureCount;
        private long totalInferenceTimeMs;
        private Instant timestamp;

        /**
         * Create a batch response from individual prediction results.
         */
        public static BatchPredictionResponse of(
                List<PredictionResponse> predictions,
                long totalInferenceTimeMs
        ) {
            int failures = (int) predictions.stream()
                    .filter(p -> p.getLabel() != null && p.getLabel().equals("error"))
                    .count();

            return BatchPredictionResponse.builder()
                    .predictions(predictions)
                    .totalRequests(predictions.size())
                    .successCount(predictions.size() - failures)
                    .failureCount(failures)
                    .totalInferenceTimeMs(totalInferenceTimeMs)
                    .timestamp(Instant.now())
                    .build();
        }
    }

    /**
     * Model metadata and information response.
     */
    @Data
    @Builder
    @NoArgsConstructor
    @AllArgsConstructor
    public static class ModelInfoResponse {

        private String modelName;
        private String modelVersion;
        private String modelPath;
        private int inputSize;
        private List<String> inputNames;
        private List<String> outputNames;
        private Map<String, String> metadata;
        private boolean loaded;
        private Instant loadedAt;
        private long totalPredictions;
        private double avgInferenceTimeMs;
    }

    /**
     * Health check response with model and service status.
     */
    @Data
    @Builder
    @NoArgsConstructor
    @AllArgsConstructor
    public static class HealthResponse {

        private String status;
        private String modelStatus;
        private String circuitBreakerStatus;
        private Instant timestamp;
        private long uptimeSeconds;
        private Map<String, Object> details;

        /**
         * Create a healthy status response.
         */
        public static HealthResponse healthy(
                String modelStatus,
                String circuitBreakerStatus,
                long uptimeSeconds
        ) {
            return HealthResponse.builder()
                    .status("UP")
                    .modelStatus(modelStatus)
                    .circuitBreakerStatus(circuitBreakerStatus)
                    .timestamp(Instant.now())
                    .uptimeSeconds(uptimeSeconds)
                    .build();
        }
    }

    /**
     * Standardized error response for API exceptions.
     */
    @Data
    @Builder
    @NoArgsConstructor
    @AllArgsConstructor
    public static class ErrorResponse {

        private int status;
        private String error;
        private String message;
        private String path;
        private Instant timestamp;

        /**
         * Create an error response from exception details.
         */
        public static ErrorResponse of(int status, String error, String message, String path) {
            return ErrorResponse.builder()
                    .status(status)
                    .error(error)
                    .message(message)
                    .path(path)
                    .timestamp(Instant.now())
                    .build();
        }
    }
}
