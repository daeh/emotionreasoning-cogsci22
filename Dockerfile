# Dockerfile

# conda executable
FROM --platform=linux/amd64 condaforge/miniforge3:4.12.0-0 as conda

# create conda env from lockfile
ADD conda-linux-64.lock /locks/conda-linux-64.lock
RUN conda create -p /opt/miniforge3/envs/envemoreas --copy --file /locks/conda-linux-64.lock
RUN conda clean --all --yes

# set base image
FROM --platform=linux/amd64 texlive/texlive:TL2021-historic

# monkeypatch python path
COPY --from=conda /opt/miniforge3/envs/envemoreas /opt/miniforge3/envs/envemoreas
RUN ln -svf /opt/miniforge3/envs/envemoreas/bin/python /usr/local/bin/python

# set the working directory in the container
WORKDIR /projimage

# copy the content of the local code directory to the working directory
COPY code code

# expose the container's python executable
### to run the executable on arbitrary code:
### > docker build --tag emotionreasoning .
### > docker run emotionreasoning {somescript.py}
### e.g. 
### > docker run -v `pwd`/:/projhost/ emotionreasoning /projhost/code/wrapper_cogsci.py -p /projhost/
ENTRYPOINT ["/opt/miniforge3/envs/envemoreas/bin/python"]

# command to run on container start
CMD ["/projimage/code/wrapper_cogsci.py", "-p", "/projhost/"]
### to run the project:
### > docker build --tag emotionreasoning .
### > docker run -v `pwd`/:/projhost/ emotionreasoning
