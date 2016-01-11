# chainer-char-rnn
karpathy's [char-rnn](https://github.com/karpathy/char-rnn) implementation by [Chainer](https://github.com/pfnet/chainer)


## Requirement
- [Chainer](https://github.com/pfnet/chainer)
```
$ pip install chainer
```

## Train
Start training the model using `train.py`, for example

```
$ python train.py
```

The `--data_dir` flag specifies the dataset to use. By default it is set to `data/tinyshakespeare` which consists of a subset of works of Shakespeare.

**Your own data**: If you'd like to use your own data create a single file `input.txt` and place it into a folder in `data/`. For example, `data/some_folder/input.txt`.



## Sampling
Given a checkpoint file (such as those written to cv) we can generate new text. For example:
```
$ python sample.py \
--vocabulary data/tinyshakespeare/vocab.bin \
--model cv/some_checkpoint.chainermodel \
--primetext some_text --gpu -1
```
## References
- Original implementation: https://github.com/karpathy/char-rnn
- Blog post: http://karpathy.github.io/2015/05/21/rnn-effectiveness/
