FROM tensorflow/tensorflow:latest-gpu-py3

RUN mkdir saved_model
RUN mkdir input
RUN mkdir output
COPY saved_model/blm_2_t_mn.h5 saved_model/blm_2_t_mn.h5
COPY predict.py predict.py
RUN pip install --upgrade pip
RUN pip install Pillow
RUN pip install scikit-learn
RUN pip install urllib3
RUN pip install matplotlib