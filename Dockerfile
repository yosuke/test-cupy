FROM cupy/cupy:v6.0.0

COPY test.py /test.py

CMD python /test.py