FROM mi2rl/mi2rl_image:201012

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update
RUN pip install scikit-image
# install keras_contrib for tensorflow.keras
RUN pip install git+https://www.github.com/keras-team/keras-contrib.git
RUN pip install sklearn




