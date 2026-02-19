package com.portfolio.mlapi.service;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.util.*;

/**
 * Input preprocessing service for ML model inference.
 *
 * Handles feature normalization, input validation, type checking,
 * and categorical encoding before passing data to the model.
 */
@Service
public class PreprocessingService {

    private static final Logger logger = LoggerFactory.getLogger(PreprocessingService.class);

    @Value("${model.input-size:10}")
    private int expectedInputSize;

    private static final Map<String, double[]> FEATURE_STATS = new HashMap<>();
    private static final Map<String, Map<String, Integer>> CATEGORICAL_ENCODINGS = new HashMap<>();

    static {
        FEATURE_STATS.put("amount", new double[]{250.0, 500.0});
        FEATURE_STATS.put("time", new double[]{43200.0, 25000.0});
        FEATURE_STATS.put("v1", new double[]{0.0, 1.0});
        FEATURE_STATS.put("v2", new double[]{0.0, 1.0});
        FEATURE_STATS.put("v3", new double[]{0.0, 1.0});
        FEATURE_STATS.put("v4", new double[]{0.0, 1.0});
        FEATURE_STATS.put("v5", new double[]{0.0, 1.0});
        FEATURE_STATS.put("v6", new double[]{0.0, 1.0});
        FEATURE_STATS.put("v7", new double[]{0.0, 1.0});
        FEATURE_STATS.put("v8", new double[]{0.0, 1.0});

        Map<String, Integer> merchantCategory = new LinkedHashMap<>();
        merchantCategory.put("retail", 0);
        merchantCategory.put("grocery", 1);
        merchantCategory.put("gas_station", 2);
        merchantCategory.put("restaurant", 3);
        merchantCategory.put("online", 4);
        merchantCategory.put("travel", 5);
        merchantCategory.put("entertainment", 6);
        merchantCategory.put("other", 7);
        CATEGORICAL_ENCODINGS.put("merchant_category", merchantCategory);

        Map<String, Integer> cardType = new LinkedHashMap<>();
        cardType.put("credit", 0);
        cardType.put("debit", 1);
        cardType.put("prepaid", 2);
        CATEGORICAL_ENCODINGS.put("card_type", cardType);
    }

    /**
     * Normalize feature values using z-score standardization.
     *
     * Applies (value - mean) / std normalization using pre-computed
     * feature statistics. Features without known statistics are
     * passed through unchanged.
     *
     * @param features     raw feature values
     * @param featureNames optional feature name mapping
     * @return normalized feature array
     */
    public float[] normalizeFeatures(List<Double> features, List<String> featureNames) {
        float[] normalized = new float[features.size()];

        for (int i = 0; i < features.size(); i++) {
            double value = features.get(i);
            String name = (featureNames != null && i < featureNames.size())
                    ? featureNames.get(i)
                    : "feature_" + i;

            if (FEATURE_STATS.containsKey(name)) {
                double[] stats = FEATURE_STATS.get(name);
                double mean = stats[0];
                double std = stats[1];

                if (std > 0) {
                    normalized[i] = (float) ((value - mean) / std);
                } else {
                    normalized[i] = (float) (value - mean);
                }
            } else {
                normalized[i] = (float) value;
            }

            if (Float.isNaN(normalized[i]) || Float.isInfinite(normalized[i])) {
                logger.warn("Invalid normalized value for feature '{}' at index {}: {}", name, i, normalized[i]);
                normalized[i] = 0.0f;
            }
        }

        return normalized;
    }

    /**
     * Validate input features for correctness and completeness.
     *
     * Checks for null values, NaN, infinity, expected size,
     * and reasonable value ranges.
     *
     * @param features raw feature values
     * @return list of validation error messages (empty if valid)
     */
    public List<String> validateInput(List<Double> features) {
        List<String> errors = new ArrayList<>();

        if (features == null) {
            errors.add("Features list must not be null");
            return errors;
        }

        if (features.isEmpty()) {
            errors.add("Features list must not be empty");
            return errors;
        }

        if (features.size() != expectedInputSize) {
            errors.add(String.format(
                    "Expected %d features but received %d",
                    expectedInputSize, features.size()
            ));
        }

        for (int i = 0; i < features.size(); i++) {
            Double value = features.get(i);

            if (value == null) {
                errors.add(String.format("Feature at index %d is null", i));
                continue;
            }

            if (Double.isNaN(value)) {
                errors.add(String.format("Feature at index %d is NaN", i));
            }

            if (Double.isInfinite(value)) {
                errors.add(String.format("Feature at index %d is infinite", i));
            }

            if (Math.abs(value) > 1e10) {
                errors.add(String.format(
                        "Feature at index %d has suspicious magnitude: %.2e", i, value
                ));
            }
        }

        return errors;
    }

    /**
     * Encode categorical features as integer values.
     *
     * Maps string category values to their corresponding integer
     * encodings. Unknown categories are mapped to a default value.
     *
     * @param categoryName the name of the categorical feature
     * @param value        the category string value
     * @return integer encoding for the category
     */
    public int encodeCategorical(String categoryName, String value) {
        Map<String, Integer> encoding = CATEGORICAL_ENCODINGS.get(categoryName);

        if (encoding == null) {
            logger.warn("Unknown categorical feature: {}", categoryName);
            return -1;
        }

        String normalizedValue = value.toLowerCase().trim();
        Integer encoded = encoding.get(normalizedValue);

        if (encoded == null) {
            logger.debug("Unknown category value '{}' for feature '{}', using default", value, categoryName);
            return encoding.getOrDefault("other", encoding.size());
        }

        return encoded;
    }

    /**
     * Get the set of known categorical feature names.
     */
    public Set<String> getCategoricalFeatures() {
        return Collections.unmodifiableSet(CATEGORICAL_ENCODINGS.keySet());
    }

    /**
     * Get the encoding map for a specific categorical feature.
     */
    public Map<String, Integer> getCategoricalEncoding(String categoryName) {
        Map<String, Integer> encoding = CATEGORICAL_ENCODINGS.get(categoryName);
        return encoding != null ? Collections.unmodifiableMap(encoding) : Collections.emptyMap();
    }

    /**
     * Get the expected input size for the model.
     */
    public int getExpectedInputSize() {
        return expectedInputSize;
    }
}
