FROM registry.gitlab.uspex-team.org/uspex_python/thirdparty:master
COPY build/dist/USPEX-*-cp36-cp36m-linux_x86_64.whl /opt/
RUN pip install USPEX-*-cp36-cp36m-linux_x86_64.whl && rm USPEX-*-cp36-cp36m-linux_x86_64.whl
