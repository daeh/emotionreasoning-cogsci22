# Dockerfile

FROM --platform=linux/amd64 debian:bullseye-slim

# Docker default of `/bin/sh` doesn't support `source`
SHELL ["/bin/bash", "-c"]

### Install conda

RUN apt-get update
RUN apt-get install --no-install-recommends --yes wget bzip2 ca-certificates git tini

RUN apt-get clean
RUN rm -rf /var/lib/apt/lists/*

RUN mkdir /imgbase
WORKDIR /imgbase

ARG MINIFORGE_NAME=Miniforge3
ARG MINIFORGE_VERSION=22.9.0-1
ARG TARGETPLATFORM

ENV CONDA_DIR=/opt/conda
# ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV PATH=${CONDA_DIR}/bin:${PATH}

RUN wget --no-hsts --quiet https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/${MINIFORGE_NAME}-${MINIFORGE_VERSION}-Linux-$(uname -m).sh -O /imgbase/miniforge.sh
# wget -O Miniforge3.sh "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"

RUN ["chmod", "+x", "/imgbase/miniforge.sh"]
RUN /imgbase/miniforge.sh -b -p ${CONDA_DIR}

RUN rm /imgbase/miniforge.sh
RUN conda clean --tarballs --index-cache --packages --yes
RUN find ${CONDA_DIR} -follow -type f -name '*.a' -delete
RUN find ${CONDA_DIR} -follow -type f -name '*.pyc' -delete
RUN conda clean --force-pkgs-dirs --all --yes
# RUN echo ". ${CONDA_DIR}/etc/profile.d/conda.sh && conda activate base" >> /etc/skel/.bashrc
# RUN echo ". ${CONDA_DIR}/etc/profile.d/conda.sh && conda activate base" >> ~/.bashrc

### Setup conda env

### Create conda env from lockfile
COPY conda-linux-64.lock /imgbase/conda-linux-64.lock
RUN conda create --name emoreasenv --file /imgbase/conda-linux-64.lock
RUN conda clean --all --yes

RUN echo ". ${CONDA_DIR}/etc/profile.d/conda.sh && conda activate emoreasenv" >> /etc/skel/.bashrc
RUN echo ". ${CONDA_DIR}/etc/profile.d/conda.sh && conda activate emoreasenv" >> ~/.bashrc

### Install TeXLive

### :latest with apt-get
RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends texlive-full
RUN apt-get clean
RUN rm -rf /var/lib/apt/lists/*

### :latest with tinytex
# RUN wget -qO- "https://yihui.org/tinytex/install-bin-unix.sh" | sh

### :latest based on https://github.com/kayhannay/efalive_docker/blob/main/Dockerfile
# ENV DEBIAN_FRONTEND noninteractive
# ENV LC_ALL=C.UTF-8
# ENV LANG=C.UTF-8

# RUN echo "deb http://cdn-fastly.deb.debian.org/debian bullseye main" > /etc/apt/sources.list && echo "deb http://cdn-fastly.deb.debian.org/debian bullseye-updates main" >> /etc/apt/sources.list && echo "deb http://security.debian.org/debian-security bullseye-security main contrib non-free" >> /etc/apt/sources.list

# RUN apt update \
#     && DEBIAN_FRONTEND=noninteractive apt install -y \
#     live-build \
#     texlive-latex-base \
#     texlive-latex-extra \
#     texlive-latex-recommended 

# RUN apt update \
#     && DEBIAN_FRONTEND=noninteractive apt install -y \
#     lftp \
#     rsync \
#     apt-cacher-ng \
#     vim \
#     git \
#     live-build \
#     texlive-lang-german \
#     texlive-latex-base \
#     texlive-latex-extra \
#     texlive-latex-recommended \
#     python3 \
#     python3-pip \
#     python3-pkgconfig \
#     docbook-to-man \
#     devscripts \
#     dpkg-dev \
#     reprepro \
#     sudo \
#     libgirepository1.0-dev \
#     libglib2.0-dev \
#     libcairo2-dev \
#     libffi-dev \
#     gir1.2-gtk-3.0 \
#     gir1.2-gudev-1.0

# Copy & setup TeXLive binary from installer
ENV PATH=/usr/local/bin/texlive:$PATH

# COPY --from=installer /usr/local/texlive /usr/local/texlive
# RUN rm -f /etc/apt/apt.conf.d/docker-clean; echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > /etc/apt/apt.conf.d/keep-cache
# RUN --mount=type=cache,target=/var/cache/apt --mount=type=cache,target=/var/lib/apt \
#     apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
#     perl \
#     wget && \
#     ln -sf /usr/local/texlive/*/bin/* /usr/local/bin/texlive

# RUN apt-get clean
# RUN rm -rf /var/lib/apt/lists/*


ENTRYPOINT ["/opt/conda/envs/emoreasenv/bin/python"]