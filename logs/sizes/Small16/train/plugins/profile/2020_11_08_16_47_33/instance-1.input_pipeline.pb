	$����?$����?!$����?	��@�j� @��@�j� @!��@�j� @"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$$����?���s�v�?A�_x%���?Y->�x�?*	�� �r�h@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate-�s�ڵ?!E�e��E@)v�d�?1�d�u�JD@:Preprocessing2F
Iterator::Model�g���?!���)
�A@)�!�uq�?1�O�qQ�<@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatN���P�?!��G)\ 0@)�I}Yک�?1��z �)@:Preprocessing2U
Iterator::Model::ParallelMapV2�O �Ȓ�?!V�ԇs@)�O �Ȓ�?1V�ԇs@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�\��$?�?!����*P@)�\R��?1�0�D^�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor���y?!O·^��	@)���y?1O·^��	@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicel{�%9`w?!W��6C@)l{�%9`w?1W��6C@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap|E�^Ӄ�?!�+��gF@)�)r��9e?1E��`�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is MODERATELY input-bound because 8.4% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*high2t38.1 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9��@�j� @>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	���s�v�?���s�v�?!���s�v�?      ��!       "      ��!       *      ��!       2	�_x%���?�_x%���?!�_x%���?:      ��!       B      ��!       J	->�x�?->�x�?!->�x�?R      ��!       Z	->�x�?->�x�?!->�x�?JCPU_ONLYY��@�j� @b 