# Basic python3 image as base
FROM harbor2.vantage6.ai/infrastructure/algorithm-base:4.2

# Change this to the package name of your project. This needs to be the same
# as what you specified for the name in the `setup.py`.
ARG PKG_NAME="v6_logistic_regression_py"

# This will install your algorithm into this image.
COPY . /app
RUN pip install /app

# This will run your algorithm when the Docker container is started.
ENV PKG_NAME=${PKG_NAME}
CMD python -c "from vantage6.algorithm.tools.wrap import wrap_algorithm; wrap_algorithm()"
