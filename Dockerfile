# Gizmo inference server - multi-stage Docker build.
# Produces a small image with the static gizmo binary and system CA certs.

FROM ubuntu:22.04 AS builder

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential cmake git ca-certificates python3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY . .

# Initialise llama.cpp submodule if it was not already populated.
RUN git submodule update --init --recursive || true

# Build a portable binary inside the container (do not target the build host CPU).
RUN cmake -S . -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_NATIVE=OFF \
        -DCMAKE_INSTALL_PREFIX=/opt/gizmo

RUN cmake --build build -j$(nproc)
RUN cmake --install build

# ---- Runtime image ----
FROM ubuntu:22.04

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/gizmo /usr/local

VOLUME ["/models"]
ENV GIZMO_MODEL_PATH=/models
EXPOSE 8080

ENTRYPOINT ["gizmo"]
CMD ["serve"]
