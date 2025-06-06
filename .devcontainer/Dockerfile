# Use Python as base
# Use Python as base
FROM python:3.12-bullseye

# Build args for your local user
ARG CURRENT_UID=1000
ARG CURRENT_GID=1000

# Create group/user in container
RUN if [ "$CURRENT_UID" = "0" ] || [ "$CURRENT_GID" = "0" ]; then \
      echo "Detected root UID/GID => fallback to UID=1000, GID=1000"; \
      groupadd -g 1000 appuser && useradd -u 1000 -g 1000 -m appuser; \
    else \
      echo "Creating user with UID=$CURRENT_UID, GID=$CURRENT_GID"; \
      groupadd -g $CURRENT_GID appuser && useradd -u $CURRENT_UID -g $CURRENT_GID -m appuser; \
    fi

RUN apt-get update \
    && apt-get install -y sudo locales \
    && sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen \
    && dpkg-reconfigure --frontend=noninteractive locales \
    && update-locale LC_ALL=en_US.UTF-8 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Give passwordless sudo if desired
RUN echo "appuser ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/appuser

# **Key**: make sure the user has a bash shell
RUN usermod --shell /bin/bash appuser

# Optionally set environment variables for your shell
ENV LC_ALL=en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US.UTF-8

# Create and install in a virtual environment
RUN mkdir -p /usr/src/venvs \
    && python -m venv /usr/src/venvs/app-main
COPY requirements.txt /usr/src/
RUN /usr/src/venvs/app-main/bin/pip install --upgrade pip \
    && /usr/src/venvs/app-main/bin/pip install -r /usr/src/requirements.txt \
    && chown -R appuser:appuser /usr/src/venvs/app-main

# Switch to the non-root user
USER appuser
WORKDIR /usr/src/project

# **Key**: default command is /bin/bash
CMD ["/bin/bash"]

