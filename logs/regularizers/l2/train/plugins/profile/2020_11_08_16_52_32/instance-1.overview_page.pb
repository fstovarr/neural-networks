�	�G�C�@�G�C�@!�G�C�@	j��y@j��y@!j��y@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$�G�C�@4�%��?A��.�`@YqW�"��?*	+�kt@2F
Iterator::Model ����?!���N@)7l[�� �?1��G�TCI@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenateۤ����?!��0 3@)	l��3��?1gta1�.@:Preprocessing2U
Iterator::Model::ParallelMapV2"�*��<�?!=���p�%@)"�*��<�?1=���p�%@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�0C� �?!C���u*@)���;3�?1�,b��$@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip,f��!�?!h�P�IC@)/�4'/2�?1U�Ǫ�@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice4�Y��U�?!��F<K@)4�Y��U�?1��F<K@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor$�P29��?!<R}���@)$�P29��?1<R}���@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap����>�?!�H�׫34@)�
�H�<m?1=�[o�z�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 9.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9k��y@>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	4�%��?4�%��?!4�%��?      ��!       "      ��!       *      ��!       2	��.�`@��.�`@!��.�`@:      ��!       B      ��!       J	qW�"��?qW�"��?!qW�"��?R      ��!       Z	qW�"��?qW�"��?!qW�"��?JCPU_ONLYYk��y@b Y      Y@q*M���#@"�
both�Your program is POTENTIALLY input-bound because 9.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQ2"CPU: B 