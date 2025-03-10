FROM condaforge/mambaforge

WORKDIR /project
USER root

COPY ENV.yml .

RUN apt-get update && apt-get install -y build-essential

RUN mamba env create -n esp_vis --file ENV.yml && \
    mamba clean --all --yes && \
    mamba init 

SHELL ["conda", "run", "-n", "esp_vis", "/bin/bash", "-c"]

RUN  git clone https://github.com/SimonBoothroyd/molesp.git && \
    cd molesp && \
    pip install -e . && \
    python setup.py build_gui && \
    python setup.py install

RUN git clone https://github.com/bismuthadams1/ChargeAPI.git && \
    cd ChargeAPI && \
    pip install -e .

RUN pip install git+https://github.com/bismuthadams1/nagl.git

SHELL ["/bin/bash", "-c"]

RUN git clone https://github.com/bismuthadams1/nagl-mbis.git && \
    cd nagl-mbis && \
    mamba env create -n naglmbis --file ./devtools/conda-envs/env.yaml && \
    mamba run -n naglmbis pip install -e . --no-build-isolation

# (Optional) If you want to install Riniker, uncomment the following:
# RUN  mamba deactivate && \
#     git clone https://github.com/kuano-ai/Forked_Riniker.git && \
#     cd Forked_Riniker && \
#     conda env create -n riniker --file riniker.yml && \
#     pip install -e .

SHELL ["conda", "run", "-n", "esp_vis", "/bin/bash", "-c"]

EXPOSE 8000

# Set the default command to run your ESP visualizer (adjust the script name/args as needed)
CMD ["python", "esp_visualize_esp.py"]
