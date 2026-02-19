FROM eclipse-temurin:17-jre-alpine AS runtime

LABEL maintainer="portfolio" \
      description="Spring Boot ML API - ONNX model serving microservice"

RUN addgroup -S appgroup && adduser -S appuser -G appgroup

WORKDIR /app

RUN mkdir -p /app/models && chown -R appuser:appgroup /app

COPY --chown=appuser:appgroup target/*.jar app.jar

USER appuser

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD wget -q --spider http://localhost:8080/actuator/health || exit 1

ENV JAVA_OPTS="-Xms256m -Xmx512m -XX:+UseG1GC -XX:MaxGCPauseMillis=100"

ENTRYPOINT ["sh", "-c", "java ${JAVA_OPTS} -jar app.jar"]
