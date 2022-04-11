FROM ignaciovizzo/vdb_to_numpy
LABEL maintainer="Ignacio Vizzo <ivizzo@uni-bonn.de>"
ENV TERM xterm
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /apps

COPY . /vdb_to_numpy
RUN cd /vdb_to_numpy && make install && cd .. && rm -rf vdb_to_numpy

COPY ./apps/ /apps
ENTRYPOINT ["python3","/apps/mesh_to_sdf.py"]

