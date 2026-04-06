# docker/arcadedb/Dockerfile
# Multi-stage build: JDK for compile, JRE for runtime

FROM maven:3.9-eclipse-temurin-21 AS builder
WORKDIR /build
COPY repo/ .
RUN mvn clean install -DskipTests -q && \
    mkdir -p /dist && \
    dist_src="$(find package/target -mindepth 2 -maxdepth 2 -type d \
      -path 'package/target/arcadedb-*.dir/arcadedb-*' \
      | grep -vE '(-minimal|-headless)$' \
      | head -n1)" && \
    test -n "$dist_src" && \
    cp -a "$dist_src"/. /dist/

FROM eclipse-temurin:21-jre
WORKDIR /home/arcadedb
COPY --from=builder /dist/ ./
VOLUME /home/arcadedb/databases
EXPOSE 2480 2424
ENTRYPOINT ["bin/server.sh"]
