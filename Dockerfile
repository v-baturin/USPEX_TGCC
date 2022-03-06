FROM registry.gitlab.uspex-team.org/uspex_python/thirdparty:master
COPY build/dist/uspex-*-cp36-cp36m-linux_x86_64.whl /opt/
RUN pip install uspex-*-cp36-cp36m-linux_x86_64.whl && rm uspex-*-cp36-cp36m-linux_x86_64.whl
