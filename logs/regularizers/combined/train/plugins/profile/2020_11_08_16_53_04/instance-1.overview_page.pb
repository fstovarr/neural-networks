�	�
cA~@�
cA~@!�
cA~@	�Q��@�Q��@!�Q��@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$�
cA~@�;��?A |(ђ�@Y�"�-�R�?*	�����q@2F
Iterator::Model�#�G�?!�1����L@)LTol��?1yU�ߊ�F@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateIJzZ��?!�{�â3@)���ި�?1�]�?O.@:Preprocessing2U
Iterator::Model::ParallelMapV2���m�?!�qKWO�(@)���m�?1�qKWO�(@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat��x]�?!��˒0@)�}��?1��ǯ�(@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipl]j�~��?! �EJaE@)��P�l�?1�i/<1@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�=%���?!4+B��@)�=%���?14+B��@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensoryx��e�?!�<4>�]@)yx��e�?1�<4>�]@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�ׁsF��?!��=���4@)0�x��no?1�<SБ�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is MODERATELY input-bound because 6.4% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*moderate2s7.6 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9�Q��@>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�;��?�;��?!�;��?      ��!       "      ��!       *      ��!       2	 |(ђ�@ |(ђ�@! |(ђ�@:      ��!       B      ��!       J	�"�-�R�?�"�-�R�?!�"�-�R�?R      ��!       Z	�"�-�R�?�"�-�R�?!�"�-�R�?JCPU_ONLYY�Q��@b Y      Y@q;��ܤn.@"�	
both�Your program is MODERATELY input-bound because 6.4% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
nomoderate"s7.6 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.:
Refer to the TF2 Profiler FAQb�15.2161% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 