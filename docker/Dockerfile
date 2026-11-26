# Image of the MGTS solution: the organizers' stand (built from ivr0ma/hackathon as "hakatonius")
# plus the Python dependency of mgts_go2 and tools for headless RViz screen recording.
#   docker build -t hakatonius <organizers' stand folder>
#   docker build -t mgts-go2 -f docker/Dockerfile .
ARG BASE=hakatonius
FROM ${BASE}

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-scipy python3-yaml \
        xvfb ffmpeg x11-utils xdotool \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /ros2_ws
