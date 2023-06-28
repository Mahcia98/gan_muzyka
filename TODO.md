# old:
- zmierzyć czas na lokalnym kompie 30min ile to epok
- ogarnij google colab
- odpal na Google Colab - sprawdź czy szybciej działa
- jeśli działa szybciej to sprawdź jakie wyniki się pojawiają po 1 epoce
- jeżeli nie - przyjrzeć modelowi
  - albo zmiany w tym tensorflow
  - albo lepsza implementacja w PyTorch-u (DCGAN) 


# 2023_06_28

- colab RAM 12->25GB - wyszukaj w google/stackoverflow
- add callbacks: https://blog.paperspace.com/tensorflow-callbacks/
```python
from datetime import datetime
current_date = datetime.today().strftime('%Y_%m_%d_%H_%M')
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
  'model_saves/model_{current_date}.ckpt',
  monitor='val_loss',
  verbose=0,
  save_best_only=True,
  save_weights_only=False,
  mode='auto',
  save_freq='epoch'
)
```
```python
csv_logger = tf.keras.callbacks.CSVLogger(
    'model_history/model_{current_date}.csv',
     separator=',', 
     append=False
)
terminate_on_nan = tf.keras.callbacks.TerminateOnNaN()
```
- stwórz model_saves, model_history
- dodaj model_saves, model_history do .gitignore
- dodaj callbacki w metodzie fit: 
```python
   model.fit(..., callbacks=[csv_logger, model_checkpoint, terminate_on_nan])
```
- save images every X batches
- Odpal ten model na Google Colab, jeżeli się wywali to zmniejsz batch_size dwukrotnie. Potrenuj go przez jakieś dwie 3 h zobaczę jakie są wyniki jeżeli te obrazki generowane przez sieć będą czarne i zaczną się pojawiać jakieś poziome paski to git. Jak nie to zmniejszamy model.

