	�0|D��?�0|D��?!�0|D��?	6L�;��!@6L�;��!@!6L�;��!@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$�0|D��?��lY�.�?AZ��ڊ��?Y���XP�?*	u�V*k@2F
Iterator::Model=E7��?!?�!�(�G@)m���{�?1�Qs�^�B@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�"�J %�?!�ͥͪN@@)�k����?1B�uvG�;@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?;�bF�?!jCP��@-@)]�@�"�?1��rk�&@:Preprocessing2U
Iterator::Model::ParallelMapV2?�=x�Җ?!�ֹ�'�$@)?�=x�Җ?1�ֹ�'�$@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceQf�L2r�?!*+W�8,@)Qf�L2r�?1*+W�8,@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip���C��?!�8�H�J@)���1>�~?10�ѹ�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor���֪}?!K�&K��
@)���֪}?1K�&K��
@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap���N�?!��o�� A@)àL���h?1��D�}C�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is MODERATELY input-bound because 8.9% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*high2t24.6 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.97L�;��!@>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��lY�.�?��lY�.�?!��lY�.�?      ��!       "      ��!       *      ��!       2	Z��ڊ��?Z��ڊ��?!Z��ڊ��?:      ��!       B      ��!       J	���XP�?���XP�?!���XP�?R      ��!       Z	���XP�?���XP�?!���XP�?JCPU_ONLYY7L�;��!@b 