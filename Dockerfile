FROM registry.gitlab.uspex-team.org/uspex_python/thirdparty:master
COPY USPEX_AGAIN-0.0.1-cp36-cp36m-linux_x86_64.whl /opt/USPEX_AGAIN-0.0.1-cp36-cp36m-linux_x86_64.whl
RUN pip install USPEX_AGAIN-0.0.1-cp36-cp36m-linux_x86_64.whl && rm USPEX_AGAIN-0.0.1-cp36-cp36m-linux_x86_64.whl
