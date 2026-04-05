# docker/arcadedb/Dockerfile
# Multi-stage build: JDK for compile, JRE for runtime

FROM eclipse-temurin:21-jdk AS builder
WORKDIR /build
COPY repo/ .
RUN ./gradlew build -x test --no-daemon -q

FROM eclipse-temurin:21-jre
WORKDIR /home/arcadedb
COPY --from=builder /build/package/target/arcadedb-*/lib/ lib/
COPY --from=builder /build/package/target/arcadedb-*/bin/ bin/
COPY --from=builder /build/package/target/arcadedb-*/config/ config/
VOLUME /home/arcadedb/databases
EXPOSE 2480 2424
ENTRYPOINT ["bin/server.sh"]
