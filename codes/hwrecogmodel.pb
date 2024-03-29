node {
  name: "input/x-input"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: -1
        }
        dim {
          size: 784
        }
      }
    }
  }
}
node {
  name: "input/y-input"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: -1
        }
        dim {
          size: 10
        }
      }
    }
  }
}
node {
  name: "input_reshape/Reshape/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\377\377\377\377\034\000\000\000\034\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "input_reshape/Reshape"
  op: "Reshape"
  input: "input/x-input"
  input: "input_reshape/Reshape/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "input_reshape/input/tag"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
        }
        string_val: "input_reshape/input"
      }
    }
  }
}
node {
  name: "input_reshape/input"
  op: "ImageSummary"
  input: "input_reshape/input/tag"
  input: "input_reshape/Reshape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "bad_color"
    value {
      tensor {
        dtype: DT_UINT8
        tensor_shape {
          dim {
            size: 4
          }
        }
        int_val: 255
        int_val: 0
        int_val: 0
        int_val: 255
      }
    }
  }
  attr {
    key: "max_images"
    value {
      i: 10
    }
  }
}
node {
  name: "layer1/weights/truncated_normal/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "\020\003\000\000\364\001\000\000"
      }
    }
  }
}
node {
  name: "layer1/weights/truncated_normal/mean"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "layer1/weights/truncated_normal/stddev"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.10000000149011612
      }
    }
  }
}
node {
  name: "layer1/weights/truncated_normal/TruncatedNormal"
  op: "TruncatedNormal"
  input: "layer1/weights/truncated_normal/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 0
    }
  }
  attr {
    key: "seed2"
    value {
      i: 0
    }
  }
}
node {
  name: "layer1/weights/truncated_normal/mul"
  op: "Mul"
  input: "layer1/weights/truncated_normal/TruncatedNormal"
  input: "layer1/weights/truncated_normal/stddev"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "layer1/weights/truncated_normal"
  op: "Add"
  input: "layer1/weights/truncated_normal/mul"
  input: "layer1/weights/truncated_normal/mean"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "layer1/weights/Variable"
  op: "VariableV2"
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 784
        }
        dim {
          size: 500
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "layer1/weights/Variable/Assign"
  op: "Assign"
  input: "layer1/weights/Variable"
  input: "layer1/weights/truncated_normal"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@layer1/weights/Variable"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "layer1/weights/Variable/read"
  op: "Identity"
  input: "layer1/weights/Variable"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@layer1/weights/Variable"
      }
    }
  }
}
node {
  name: "layer1/weights/summaries/Rank"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 2
      }
    }
  }
}
node {
  name: "layer1/weights/summaries/range/start"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 0
      }
    }
  }
}
node {
  name: "layer1/weights/summaries/range/delta"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "layer1/weights/summaries/range"
  op: "Range"
  input: "layer1/weights/summaries/range/start"
  input: "layer1/weights/summaries/Rank"
  input: "layer1/weights/summaries/range/delta"
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "layer1/weights/summaries/Mean"
  op: "Mean"
  input: "layer1/weights/Variable/read"
  input: "layer1/weights/summaries/range"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "layer1/weights/summaries/mean/tags"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
        }
        string_val: "layer1/weights/summaries/mean"
      }
    }
  }
}
node {
  name: "layer1/weights/summaries/mean"
  op: "ScalarSummary"
  input: "layer1/weights/summaries/mean/tags"
  input: "layer1/weights/summaries/Mean"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "layer1/weights/summaries/stddev/sub"
  op: "Sub"
  input: "layer1/weights/Variable/read"
  input: "layer1/weights/summaries/Mean"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "layer1/weights/summaries/stddev/Square"
  op: "Square"
  input: "layer1/weights/summaries/stddev/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "layer1/weights/summaries/stddev/Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "\000\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "layer1/weights/summaries/stddev/Mean"
  op: "Mean"
  input: "layer1/weights/summaries/stddev/Square"
  input: "layer1/weights/summaries/stddev/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "layer1/weights/summaries/stddev/Sqrt"
  op: "Sqrt"
  input: "layer1/weights/summaries/stddev/Mean"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "layer1/weights/summaries/stddev_1/tags"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
        }
        string_val: "layer1/weights/summaries/stddev_1"
      }
    }
  }
}
node {
  name: "layer1/weights/summaries/stddev_1"
  op: "ScalarSummary"
  input: "layer1/weights/summaries/stddev_1/tags"
  input: "layer1/weights/summaries/stddev/Sqrt"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "layer1/weights/summaries/Rank_1"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 2
      }
    }
  }
}
node {
  name: "layer1/weights/summaries/range_1/start"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 0
      }
    }
  }
}
node {
  name: "layer1/weights/summaries/range_1/delta"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "layer1/weights/summaries/range_1"
  op: "Range"
  input: "layer1/weights/summaries/range_1/start"
  input: "layer1/weights/summaries/Rank_1"
  input: "layer1/weights/summaries/range_1/delta"
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "layer1/weights/summaries/Max"
  op: "Max"
  input: "layer1/weights/Variable/read"
  input: "layer1/weights/summaries/range_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "layer1/weights/summaries/max/tags"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
        }
        string_val: "layer1/weights/summaries/max"
      }
    }
  }
}
node {
  name: "layer1/weights/summaries/max"
  op: "ScalarSummary"
  input: "layer1/weights/summaries/max/tags"
  input: "layer1/weights/summaries/Max"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "layer1/weights/summaries/Rank_2"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 2
      }
    }
  }
}
node {
  name: "layer1/weights/summaries/range_2/start"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 0
      }
    }
  }
}
node {
  name: "layer1/weights/summaries/range_2/delta"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "layer1/weights/summaries/range_2"
  op: "Range"
  input: "layer1/weights/summaries/range_2/start"
  input: "layer1/weights/summaries/Rank_2"
  input: "layer1/weights/summaries/range_2/delta"
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "layer1/weights/summaries/Min"
  op: "Min"
  input: "layer1/weights/Variable/read"
  input: "layer1/weights/summaries/range_2"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "layer1/weights/summaries/min/tags"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
        }
        string_val: "layer1/weights/summaries/min"
      }
    }
  }
}
node {
  name: "layer1/weights/summaries/min"
  op: "ScalarSummary"
  input: "layer1/weights/summaries/min/tags"
  input: "layer1/weights/summaries/Min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "layer1/biases/Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 500
          }
        }
        float_val: 0.10000000149011612
      }
    }
  }
}
node {
  name: "layer1/biases/Variable"
  op: "VariableV2"
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 500
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "layer1/biases/Variable/Assign"
  op: "Assign"
  input: "layer1/biases/Variable"
  input: "layer1/biases/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@layer1/biases/Variable"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "layer1/biases/Variable/read"
  op: "Identity"
  input: "layer1/biases/Variable"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@layer1/biases/Variable"
      }
    }
  }
}
node {
  name: "layer1/biases/summaries/Rank"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "layer1/biases/summaries/range/start"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 0
      }
    }
  }
}
node {
  name: "layer1/biases/summaries/range/delta"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "layer1/biases/summaries/range"
  op: "Range"
  input: "layer1/biases/summaries/range/start"
  input: "layer1/biases/summaries/Rank"
  input: "layer1/biases/summaries/range/delta"
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "layer1/biases/summaries/Mean"
  op: "Mean"
  input: "layer1/biases/Variable/read"
  input: "layer1/biases/summaries/range"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "layer1/biases/summaries/mean/tags"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
        }
        string_val: "layer1/biases/summaries/mean"
      }
    }
  }
}
node {
  name: "layer1/biases/summaries/mean"
  op: "ScalarSummary"
  input: "layer1/biases/summaries/mean/tags"
  input: "layer1/biases/summaries/Mean"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "layer1/biases/summaries/stddev/sub"
  op: "Sub"
  input: "layer1/biases/Variable/read"
  input: "layer1/biases/summaries/Mean"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "layer1/biases/summaries/stddev/Square"
  op: "Square"
  input: "layer1/biases/summaries/stddev/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "layer1/biases/summaries/stddev/Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 0
      }
    }
  }
}
node {
  name: "layer1/biases/summaries/stddev/Mean"
  op: "Mean"
  input: "layer1/biases/summaries/stddev/Square"
  input: "layer1/biases/summaries/stddev/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "layer1/biases/summaries/stddev/Sqrt"
  op: "Sqrt"
  input: "layer1/biases/summaries/stddev/Mean"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "layer1/biases/summaries/stddev_1/tags"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
        }
        string_val: "layer1/biases/summaries/stddev_1"
      }
    }
  }
}
node {
  name: "layer1/biases/summaries/stddev_1"
  op: "ScalarSummary"
  input: "layer1/biases/summaries/stddev_1/tags"
  input: "layer1/biases/summaries/stddev/Sqrt"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "layer1/biases/summaries/Rank_1"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "layer1/biases/summaries/range_1/start"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 0
      }
    }
  }
}
node {
  name: "layer1/biases/summaries/range_1/delta"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "layer1/biases/summaries/range_1"
  op: "Range"
  input: "layer1/biases/summaries/range_1/start"
  input: "layer1/biases/summaries/Rank_1"
  input: "layer1/biases/summaries/range_1/delta"
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "layer1/biases/summaries/Max"
  op: "Max"
  input: "layer1/biases/Variable/read"
  input: "layer1/biases/summaries/range_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "layer1/biases/summaries/max/tags"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
        }
        string_val: "layer1/biases/summaries/max"
      }
    }
  }
}
node {
  name: "layer1/biases/summaries/max"
  op: "ScalarSummary"
  input: "layer1/biases/summaries/max/tags"
  input: "layer1/biases/summaries/Max"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "layer1/biases/summaries/Rank_2"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "layer1/biases/summaries/range_2/start"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 0
      }
    }
  }
}
node {
  name: "layer1/biases/summaries/range_2/delta"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "layer1/biases/summaries/range_2"
  op: "Range"
  input: "layer1/biases/summaries/range_2/start"
  input: "layer1/biases/summaries/Rank_2"
  input: "layer1/biases/summaries/range_2/delta"
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "layer1/biases/summaries/Min"
  op: "Min"
  input: "layer1/biases/Variable/read"
  input: "layer1/biases/summaries/range_2"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "layer1/biases/summaries/min/tags"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
        }
        string_val: "layer1/biases/summaries/min"
      }
    }
  }
}
node {
  name: "layer1/biases/summaries/min"
  op: "ScalarSummary"
  input: "layer1/biases/summaries/min/tags"
  input: "layer1/biases/summaries/Min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "layer1/linear_compute/MatMul"
  op: "MatMul"
  input: "input/x-input"
  input: "layer1/weights/Variable/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "transpose_a"
    value {
      b: false
    }
  }
  attr {
    key: "transpose_b"
    value {
      b: false
    }
  }
}
node {
  name: "layer1/linear_compute/add"
  op: "Add"
  input: "layer1/linear_compute/MatMul"
  input: "layer1/biases/Variable/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "layer1/activation"
  op: "Relu"
  input: "layer1/linear_compute/add"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "dropout/Placeholder"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        unknown_rank: true
      }
    }
  }
}
node {
  name: "dropout/dropout_keep_probability/tags"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
        }
        string_val: "dropout/dropout_keep_probability"
      }
    }
  }
}
node {
  name: "dropout/dropout_keep_probability"
  op: "ScalarSummary"
  input: "dropout/dropout_keep_probability/tags"
  input: "dropout/Placeholder"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "dropout/dropout/Shape"
  op: "Shape"
  input: "layer1/activation"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "dropout/dropout/random_uniform/min"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "dropout/dropout/random_uniform/max"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "dropout/dropout/random_uniform/RandomUniform"
  op: "RandomUniform"
  input: "dropout/dropout/Shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 0
    }
  }
  attr {
    key: "seed2"
    value {
      i: 0
    }
  }
}
node {
  name: "dropout/dropout/random_uniform/sub"
  op: "Sub"
  input: "dropout/dropout/random_uniform/max"
  input: "dropout/dropout/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "dropout/dropout/random_uniform/mul"
  op: "Mul"
  input: "dropout/dropout/random_uniform/RandomUniform"
  input: "dropout/dropout/random_uniform/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "dropout/dropout/random_uniform"
  op: "Add"
  input: "dropout/dropout/random_uniform/mul"
  input: "dropout/dropout/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "dropout/dropout/add"
  op: "Add"
  input: "dropout/Placeholder"
  input: "dropout/dropout/random_uniform"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "dropout/dropout/Floor"
  op: "Floor"
  input: "dropout/dropout/add"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "dropout/dropout/div"
  op: "RealDiv"
  input: "layer1/activation"
  input: "dropout/Placeholder"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "dropout/dropout/mul"
  op: "Mul"
  input: "dropout/dropout/div"
  input: "dropout/dropout/Floor"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "layer2/weights/truncated_normal/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "\364\001\000\000\n\000\000\000"
      }
    }
  }
}
node {
  name: "layer2/weights/truncated_normal/mean"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "layer2/weights/truncated_normal/stddev"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.10000000149011612
      }
    }
  }
}
node {
  name: "layer2/weights/truncated_normal/TruncatedNormal"
  op: "TruncatedNormal"
  input: "layer2/weights/truncated_normal/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 0
    }
  }
  attr {
    key: "seed2"
    value {
      i: 0
    }
  }
}
node {
  name: "layer2/weights/truncated_normal/mul"
  op: "Mul"
  input: "layer2/weights/truncated_normal/TruncatedNormal"
  input: "layer2/weights/truncated_normal/stddev"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "layer2/weights/truncated_normal"
  op: "Add"
  input: "layer2/weights/truncated_normal/mul"
  input: "layer2/weights/truncated_normal/mean"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "layer2/weights/Variable"
  op: "VariableV2"
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 500
        }
        dim {
          size: 10
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "layer2/weights/Variable/Assign"
  op: "Assign"
  input: "layer2/weights/Variable"
  input: "layer2/weights/truncated_normal"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@layer2/weights/Variable"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "layer2/weights/Variable/read"
  op: "Identity"
  input: "layer2/weights/Variable"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@layer2/weights/Variable"
      }
    }
  }
}
node {
  name: "layer2/weights/summaries/Rank"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 2
      }
    }
  }
}
node {
  name: "layer2/weights/summaries/range/start"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 0
      }
    }
  }
}
node {
  name: "layer2/weights/summaries/range/delta"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "layer2/weights/summaries/range"
  op: "Range"
  input: "layer2/weights/summaries/range/start"
  input: "layer2/weights/summaries/Rank"
  input: "layer2/weights/summaries/range/delta"
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "layer2/weights/summaries/Mean"
  op: "Mean"
  input: "layer2/weights/Variable/read"
  input: "layer2/weights/summaries/range"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "layer2/weights/summaries/mean/tags"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
        }
        string_val: "layer2/weights/summaries/mean"
      }
    }
  }
}
node {
  name: "layer2/weights/summaries/mean"
  op: "ScalarSummary"
  input: "layer2/weights/summaries/mean/tags"
  input: "layer2/weights/summaries/Mean"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "layer2/weights/summaries/stddev/sub"
  op: "Sub"
  input: "layer2/weights/Variable/read"
  input: "layer2/weights/summaries/Mean"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "layer2/weights/summaries/stddev/Square"
  op: "Square"
  input: "layer2/weights/summaries/stddev/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "layer2/weights/summaries/stddev/Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "\000\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "layer2/weights/summaries/stddev/Mean"
  op: "Mean"
  input: "layer2/weights/summaries/stddev/Square"
  input: "layer2/weights/summaries/stddev/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "layer2/weights/summaries/stddev/Sqrt"
  op: "Sqrt"
  input: "layer2/weights/summaries/stddev/Mean"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "layer2/weights/summaries/stddev_1/tags"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
        }
        string_val: "layer2/weights/summaries/stddev_1"
      }
    }
  }
}
node {
  name: "layer2/weights/summaries/stddev_1"
  op: "ScalarSummary"
  input: "layer2/weights/summaries/stddev_1/tags"
  input: "layer2/weights/summaries/stddev/Sqrt"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "layer2/weights/summaries/Rank_1"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 2
      }
    }
  }
}
node {
  name: "layer2/weights/summaries/range_1/start"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 0
      }
    }
  }
}
node {
  name: "layer2/weights/summaries/range_1/delta"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "layer2/weights/summaries/range_1"
  op: "Range"
  input: "layer2/weights/summaries/range_1/start"
  input: "layer2/weights/summaries/Rank_1"
  input: "layer2/weights/summaries/range_1/delta"
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "layer2/weights/summaries/Max"
  op: "Max"
  input: "layer2/weights/Variable/read"
  input: "layer2/weights/summaries/range_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "layer2/weights/summaries/max/tags"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
        }
        string_val: "layer2/weights/summaries/max"
      }
    }
  }
}
node {
  name: "layer2/weights/summaries/max"
  op: "ScalarSummary"
  input: "layer2/weights/summaries/max/tags"
  input: "layer2/weights/summaries/Max"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "layer2/weights/summaries/Rank_2"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 2
      }
    }
  }
}
node {
  name: "layer2/weights/summaries/range_2/start"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 0
      }
    }
  }
}
node {
  name: "layer2/weights/summaries/range_2/delta"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "layer2/weights/summaries/range_2"
  op: "Range"
  input: "layer2/weights/summaries/range_2/start"
  input: "layer2/weights/summaries/Rank_2"
  input: "layer2/weights/summaries/range_2/delta"
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "layer2/weights/summaries/Min"
  op: "Min"
  input: "layer2/weights/Variable/read"
  input: "layer2/weights/summaries/range_2"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "layer2/weights/summaries/min/tags"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
        }
        string_val: "layer2/weights/summaries/min"
      }
    }
  }
}
node {
  name: "layer2/weights/summaries/min"
  op: "ScalarSummary"
  input: "layer2/weights/summaries/min/tags"
  input: "layer2/weights/summaries/Min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "layer2/biases/Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 10
          }
        }
        float_val: 0.10000000149011612
      }
    }
  }
}
node {
  name: "layer2/biases/Variable"
  op: "VariableV2"
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 10
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "layer2/biases/Variable/Assign"
  op: "Assign"
  input: "layer2/biases/Variable"
  input: "layer2/biases/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@layer2/biases/Variable"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "layer2/biases/Variable/read"
  op: "Identity"
  input: "layer2/biases/Variable"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@layer2/biases/Variable"
      }
    }
  }
}
node {
  name: "layer2/biases/summaries/Rank"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "layer2/biases/summaries/range/start"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 0
      }
    }
  }
}
node {
  name: "layer2/biases/summaries/range/delta"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "layer2/biases/summaries/range"
  op: "Range"
  input: "layer2/biases/summaries/range/start"
  input: "layer2/biases/summaries/Rank"
  input: "layer2/biases/summaries/range/delta"
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "layer2/biases/summaries/Mean"
  op: "Mean"
  input: "layer2/biases/Variable/read"
  input: "layer2/biases/summaries/range"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "layer2/biases/summaries/mean/tags"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
        }
        string_val: "layer2/biases/summaries/mean"
      }
    }
  }
}
node {
  name: "layer2/biases/summaries/mean"
  op: "ScalarSummary"
  input: "layer2/biases/summaries/mean/tags"
  input: "layer2/biases/summaries/Mean"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "layer2/biases/summaries/stddev/sub"
  op: "Sub"
  input: "layer2/biases/Variable/read"
  input: "layer2/biases/summaries/Mean"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "layer2/biases/summaries/stddev/Square"
  op: "Square"
  input: "layer2/biases/summaries/stddev/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "layer2/biases/summaries/stddev/Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 0
      }
    }
  }
}
node {
  name: "layer2/biases/summaries/stddev/Mean"
  op: "Mean"
  input: "layer2/biases/summaries/stddev/Square"
  input: "layer2/biases/summaries/stddev/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "layer2/biases/summaries/stddev/Sqrt"
  op: "Sqrt"
  input: "layer2/biases/summaries/stddev/Mean"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "layer2/biases/summaries/stddev_1/tags"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
        }
        string_val: "layer2/biases/summaries/stddev_1"
      }
    }
  }
}
node {
  name: "layer2/biases/summaries/stddev_1"
  op: "ScalarSummary"
  input: "layer2/biases/summaries/stddev_1/tags"
  input: "layer2/biases/summaries/stddev/Sqrt"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "layer2/biases/summaries/Rank_1"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "layer2/biases/summaries/range_1/start"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 0
      }
    }
  }
}
node {
  name: "layer2/biases/summaries/range_1/delta"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "layer2/biases/summaries/range_1"
  op: "Range"
  input: "layer2/biases/summaries/range_1/start"
  input: "layer2/biases/summaries/Rank_1"
  input: "layer2/biases/summaries/range_1/delta"
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "layer2/biases/summaries/Max"
  op: "Max"
  input: "layer2/biases/Variable/read"
  input: "layer2/biases/summaries/range_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "layer2/biases/summaries/max/tags"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
        }
        string_val: "layer2/biases/summaries/max"
      }
    }
  }
}
node {
  name: "layer2/biases/summaries/max"
  op: "ScalarSummary"
  input: "layer2/biases/summaries/max/tags"
  input: "layer2/biases/summaries/Max"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "layer2/biases/summaries/Rank_2"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "layer2/biases/summaries/range_2/start"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 0
      }
    }
  }
}
node {
  name: "layer2/biases/summaries/range_2/delta"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "layer2/biases/summaries/range_2"
  op: "Range"
  input: "layer2/biases/summaries/range_2/start"
  input: "layer2/biases/summaries/Rank_2"
  input: "layer2/biases/summaries/range_2/delta"
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "layer2/biases/summaries/Min"
  op: "Min"
  input: "layer2/biases/Variable/read"
  input: "layer2/biases/summaries/range_2"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "layer2/biases/summaries/min/tags"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
        }
        string_val: "layer2/biases/summaries/min"
      }
    }
  }
}
node {
  name: "layer2/biases/summaries/min"
  op: "ScalarSummary"
  input: "layer2/biases/summaries/min/tags"
  input: "layer2/biases/summaries/Min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "layer2/linear_compute/MatMul"
  op: "MatMul"
  input: "dropout/dropout/mul"
  input: "layer2/weights/Variable/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "transpose_a"
    value {
      b: false
    }
  }
  attr {
    key: "transpose_b"
    value {
      b: false
    }
  }
}
node {
  name: "layer2/linear_compute/add"
  op: "Add"
  input: "layer2/linear_compute/MatMul"
  input: "layer2/biases/Variable/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "layer2/activation"
  op: "Identity"
  input: "layer2/linear_compute/add"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "loss/Rank"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 2
      }
    }
  }
}
node {
  name: "loss/Shape"
  op: "Shape"
  input: "layer2/activation"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "loss/Rank_1"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 2
      }
    }
  }
}
node {
  name: "loss/Shape_1"
  op: "Shape"
  input: "layer2/activation"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "loss/Sub/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "loss/Sub"
  op: "Sub"
  input: "loss/Rank_1"
  input: "loss/Sub/y"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "loss/Slice/begin"
  op: "Pack"
  input: "loss/Sub"
  attr {
    key: "N"
    value {
      i: 1
    }
  }
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "axis"
    value {
      i: 0
    }
  }
}
node {
  name: "loss/Slice/size"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "loss/Slice"
  op: "Slice"
  input: "loss/Shape_1"
  input: "loss/Slice/begin"
  input: "loss/Slice/size"
  attr {
    key: "Index"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "loss/concat/values_0"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: -1
      }
    }
  }
}
node {
  name: "loss/concat/axis"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 0
      }
    }
  }
}
node {
  name: "loss/concat"
  op: "ConcatV2"
  input: "loss/concat/values_0"
  input: "loss/Slice"
  input: "loss/concat/axis"
  attr {
    key: "N"
    value {
      i: 2
    }
  }
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "loss/Reshape"
  op: "Reshape"
  input: "layer2/activation"
  input: "loss/concat"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "loss/Rank_2"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 2
      }
    }
  }
}
node {
  name: "loss/Shape_2"
  op: "Shape"
  input: "input/y-input"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "loss/Sub_1/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "loss/Sub_1"
  op: "Sub"
  input: "loss/Rank_2"
  input: "loss/Sub_1/y"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "loss/Slice_1/begin"
  op: "Pack"
  input: "loss/Sub_1"
  attr {
    key: "N"
    value {
      i: 1
    }
  }
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "axis"
    value {
      i: 0
    }
  }
}
node {
  name: "loss/Slice_1/size"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "loss/Slice_1"
  op: "Slice"
  input: "loss/Shape_2"
  input: "loss/Slice_1/begin"
  input: "loss/Slice_1/size"
  attr {
    key: "Index"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "loss/concat_1/values_0"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: -1
      }
    }
  }
}
node {
  name: "loss/concat_1/axis"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 0
      }
    }
  }
}
node {
  name: "loss/concat_1"
  op: "ConcatV2"
  input: "loss/concat_1/values_0"
  input: "loss/Slice_1"
  input: "loss/concat_1/axis"
  attr {
    key: "N"
    value {
      i: 2
    }
  }
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "loss/Reshape_1"
  op: "Reshape"
  input: "input/y-input"
  input: "loss/concat_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "loss/SoftmaxCrossEntropyWithLogits"
  op: "SoftmaxCrossEntropyWithLogits"
  input: "loss/Reshape"
  input: "loss/Reshape_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "loss/Sub_2/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "loss/Sub_2"
  op: "Sub"
  input: "loss/Rank"
  input: "loss/Sub_2/y"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "loss/Slice_2/begin"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 0
      }
    }
  }
}
node {
  name: "loss/Slice_2/size"
  op: "Pack"
  input: "loss/Sub_2"
  attr {
    key: "N"
    value {
      i: 1
    }
  }
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "axis"
    value {
      i: 0
    }
  }
}
node {
  name: "loss/Slice_2"
  op: "Slice"
  input: "loss/Shape"
  input: "loss/Slice_2/begin"
  input: "loss/Slice_2/size"
  attr {
    key: "Index"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "loss/Reshape_2"
  op: "Reshape"
  input: "loss/SoftmaxCrossEntropyWithLogits"
  input: "loss/Slice_2"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "loss/total/Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 0
      }
    }
  }
}
node {
  name: "loss/total/Mean"
  op: "Mean"
  input: "loss/Reshape_2"
  input: "loss/total/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "loss/loss/tags"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
        }
        string_val: "loss/loss"
      }
    }
  }
}
node {
  name: "loss/loss"
  op: "ScalarSummary"
  input: "loss/loss/tags"
  input: "loss/total/Mean"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "train/gradients/Shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
          }
        }
      }
    }
  }
}
node {
  name: "train/gradients/Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "train/gradients/Fill"
  op: "Fill"
  input: "train/gradients/Shape"
  input: "train/gradients/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "train/gradients/loss/total/Mean_grad/Reshape/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "train/gradients/loss/total/Mean_grad/Reshape"
  op: "Reshape"
  input: "train/gradients/Fill"
  input: "train/gradients/loss/total/Mean_grad/Reshape/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/gradients/loss/total/Mean_grad/Shape"
  op: "Shape"
  input: "loss/Reshape_2"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/gradients/loss/total/Mean_grad/Tile"
  op: "Tile"
  input: "train/gradients/loss/total/Mean_grad/Reshape"
  input: "train/gradients/loss/total/Mean_grad/Shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tmultiples"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/gradients/loss/total/Mean_grad/Shape_1"
  op: "Shape"
  input: "loss/Reshape_2"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/gradients/loss/total/Mean_grad/Shape_2"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
          }
        }
      }
    }
  }
}
node {
  name: "train/gradients/loss/total/Mean_grad/Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 0
      }
    }
  }
}
node {
  name: "train/gradients/loss/total/Mean_grad/Prod"
  op: "Prod"
  input: "train/gradients/loss/total/Mean_grad/Shape_1"
  input: "train/gradients/loss/total/Mean_grad/Const"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "train/gradients/loss/total/Mean_grad/Const_1"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 0
      }
    }
  }
}
node {
  name: "train/gradients/loss/total/Mean_grad/Prod_1"
  op: "Prod"
  input: "train/gradients/loss/total/Mean_grad/Shape_2"
  input: "train/gradients/loss/total/Mean_grad/Const_1"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "train/gradients/loss/total/Mean_grad/Maximum/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "train/gradients/loss/total/Mean_grad/Maximum"
  op: "Maximum"
  input: "train/gradients/loss/total/Mean_grad/Prod_1"
  input: "train/gradients/loss/total/Mean_grad/Maximum/y"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/gradients/loss/total/Mean_grad/floordiv"
  op: "FloorDiv"
  input: "train/gradients/loss/total/Mean_grad/Prod"
  input: "train/gradients/loss/total/Mean_grad/Maximum"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/gradients/loss/total/Mean_grad/Cast"
  op: "Cast"
  input: "train/gradients/loss/total/Mean_grad/floordiv"
  attr {
    key: "DstT"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "SrcT"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/gradients/loss/total/Mean_grad/truediv"
  op: "RealDiv"
  input: "train/gradients/loss/total/Mean_grad/Tile"
  input: "train/gradients/loss/total/Mean_grad/Cast"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "train/gradients/loss/Reshape_2_grad/Shape"
  op: "Shape"
  input: "loss/SoftmaxCrossEntropyWithLogits"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/gradients/loss/Reshape_2_grad/Reshape"
  op: "Reshape"
  input: "train/gradients/loss/total/Mean_grad/truediv"
  input: "train/gradients/loss/Reshape_2_grad/Shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/gradients/zeros_like"
  op: "ZerosLike"
  input: "loss/SoftmaxCrossEntropyWithLogits:1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "train/gradients/loss/SoftmaxCrossEntropyWithLogits_grad/ExpandDims/dim"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: -1
      }
    }
  }
}
node {
  name: "train/gradients/loss/SoftmaxCrossEntropyWithLogits_grad/ExpandDims"
  op: "ExpandDims"
  input: "train/gradients/loss/Reshape_2_grad/Reshape"
  input: "train/gradients/loss/SoftmaxCrossEntropyWithLogits_grad/ExpandDims/dim"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tdim"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/gradients/loss/SoftmaxCrossEntropyWithLogits_grad/mul"
  op: "Mul"
  input: "train/gradients/loss/SoftmaxCrossEntropyWithLogits_grad/ExpandDims"
  input: "loss/SoftmaxCrossEntropyWithLogits:1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "train/gradients/loss/Reshape_grad/Shape"
  op: "Shape"
  input: "layer2/activation"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/gradients/loss/Reshape_grad/Reshape"
  op: "Reshape"
  input: "train/gradients/loss/SoftmaxCrossEntropyWithLogits_grad/mul"
  input: "train/gradients/loss/Reshape_grad/Shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/gradients/layer2/linear_compute/add_grad/Shape"
  op: "Shape"
  input: "layer2/linear_compute/MatMul"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/gradients/layer2/linear_compute/add_grad/Shape_1"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 10
      }
    }
  }
}
node {
  name: "train/gradients/layer2/linear_compute/add_grad/BroadcastGradientArgs"
  op: "BroadcastGradientArgs"
  input: "train/gradients/layer2/linear_compute/add_grad/Shape"
  input: "train/gradients/layer2/linear_compute/add_grad/Shape_1"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/gradients/layer2/linear_compute/add_grad/Sum"
  op: "Sum"
  input: "train/gradients/loss/Reshape_grad/Reshape"
  input: "train/gradients/layer2/linear_compute/add_grad/BroadcastGradientArgs"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "train/gradients/layer2/linear_compute/add_grad/Reshape"
  op: "Reshape"
  input: "train/gradients/layer2/linear_compute/add_grad/Sum"
  input: "train/gradients/layer2/linear_compute/add_grad/Shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/gradients/layer2/linear_compute/add_grad/Sum_1"
  op: "Sum"
  input: "train/gradients/loss/Reshape_grad/Reshape"
  input: "train/gradients/layer2/linear_compute/add_grad/BroadcastGradientArgs:1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "train/gradients/layer2/linear_compute/add_grad/Reshape_1"
  op: "Reshape"
  input: "train/gradients/layer2/linear_compute/add_grad/Sum_1"
  input: "train/gradients/layer2/linear_compute/add_grad/Shape_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/gradients/layer2/linear_compute/add_grad/tuple/group_deps"
  op: "NoOp"
  input: "^train/gradients/layer2/linear_compute/add_grad/Reshape"
  input: "^train/gradients/layer2/linear_compute/add_grad/Reshape_1"
}
node {
  name: "train/gradients/layer2/linear_compute/add_grad/tuple/control_dependency"
  op: "Identity"
  input: "train/gradients/layer2/linear_compute/add_grad/Reshape"
  input: "^train/gradients/layer2/linear_compute/add_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/gradients/layer2/linear_compute/add_grad/Reshape"
      }
    }
  }
}
node {
  name: "train/gradients/layer2/linear_compute/add_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "train/gradients/layer2/linear_compute/add_grad/Reshape_1"
  input: "^train/gradients/layer2/linear_compute/add_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/gradients/layer2/linear_compute/add_grad/Reshape_1"
      }
    }
  }
}
node {
  name: "train/gradients/layer2/linear_compute/MatMul_grad/MatMul"
  op: "MatMul"
  input: "train/gradients/layer2/linear_compute/add_grad/tuple/control_dependency"
  input: "layer2/weights/Variable/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "transpose_a"
    value {
      b: false
    }
  }
  attr {
    key: "transpose_b"
    value {
      b: true
    }
  }
}
node {
  name: "train/gradients/layer2/linear_compute/MatMul_grad/MatMul_1"
  op: "MatMul"
  input: "dropout/dropout/mul"
  input: "train/gradients/layer2/linear_compute/add_grad/tuple/control_dependency"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "transpose_a"
    value {
      b: true
    }
  }
  attr {
    key: "transpose_b"
    value {
      b: false
    }
  }
}
node {
  name: "train/gradients/layer2/linear_compute/MatMul_grad/tuple/group_deps"
  op: "NoOp"
  input: "^train/gradients/layer2/linear_compute/MatMul_grad/MatMul"
  input: "^train/gradients/layer2/linear_compute/MatMul_grad/MatMul_1"
}
node {
  name: "train/gradients/layer2/linear_compute/MatMul_grad/tuple/control_dependency"
  op: "Identity"
  input: "train/gradients/layer2/linear_compute/MatMul_grad/MatMul"
  input: "^train/gradients/layer2/linear_compute/MatMul_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/gradients/layer2/linear_compute/MatMul_grad/MatMul"
      }
    }
  }
}
node {
  name: "train/gradients/layer2/linear_compute/MatMul_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "train/gradients/layer2/linear_compute/MatMul_grad/MatMul_1"
  input: "^train/gradients/layer2/linear_compute/MatMul_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/gradients/layer2/linear_compute/MatMul_grad/MatMul_1"
      }
    }
  }
}
node {
  name: "train/gradients/dropout/dropout/mul_grad/Shape"
  op: "Shape"
  input: "dropout/dropout/div"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/gradients/dropout/dropout/mul_grad/Shape_1"
  op: "Shape"
  input: "dropout/dropout/Floor"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/gradients/dropout/dropout/mul_grad/BroadcastGradientArgs"
  op: "BroadcastGradientArgs"
  input: "train/gradients/dropout/dropout/mul_grad/Shape"
  input: "train/gradients/dropout/dropout/mul_grad/Shape_1"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/gradients/dropout/dropout/mul_grad/mul"
  op: "Mul"
  input: "train/gradients/layer2/linear_compute/MatMul_grad/tuple/control_dependency"
  input: "dropout/dropout/Floor"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "train/gradients/dropout/dropout/mul_grad/Sum"
  op: "Sum"
  input: "train/gradients/dropout/dropout/mul_grad/mul"
  input: "train/gradients/dropout/dropout/mul_grad/BroadcastGradientArgs"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "train/gradients/dropout/dropout/mul_grad/Reshape"
  op: "Reshape"
  input: "train/gradients/dropout/dropout/mul_grad/Sum"
  input: "train/gradients/dropout/dropout/mul_grad/Shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/gradients/dropout/dropout/mul_grad/mul_1"
  op: "Mul"
  input: "dropout/dropout/div"
  input: "train/gradients/layer2/linear_compute/MatMul_grad/tuple/control_dependency"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "train/gradients/dropout/dropout/mul_grad/Sum_1"
  op: "Sum"
  input: "train/gradients/dropout/dropout/mul_grad/mul_1"
  input: "train/gradients/dropout/dropout/mul_grad/BroadcastGradientArgs:1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "train/gradients/dropout/dropout/mul_grad/Reshape_1"
  op: "Reshape"
  input: "train/gradients/dropout/dropout/mul_grad/Sum_1"
  input: "train/gradients/dropout/dropout/mul_grad/Shape_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/gradients/dropout/dropout/mul_grad/tuple/group_deps"
  op: "NoOp"
  input: "^train/gradients/dropout/dropout/mul_grad/Reshape"
  input: "^train/gradients/dropout/dropout/mul_grad/Reshape_1"
}
node {
  name: "train/gradients/dropout/dropout/mul_grad/tuple/control_dependency"
  op: "Identity"
  input: "train/gradients/dropout/dropout/mul_grad/Reshape"
  input: "^train/gradients/dropout/dropout/mul_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/gradients/dropout/dropout/mul_grad/Reshape"
      }
    }
  }
}
node {
  name: "train/gradients/dropout/dropout/mul_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "train/gradients/dropout/dropout/mul_grad/Reshape_1"
  input: "^train/gradients/dropout/dropout/mul_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/gradients/dropout/dropout/mul_grad/Reshape_1"
      }
    }
  }
}
node {
  name: "train/gradients/dropout/dropout/div_grad/Shape"
  op: "Shape"
  input: "layer1/activation"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/gradients/dropout/dropout/div_grad/Shape_1"
  op: "Shape"
  input: "dropout/Placeholder"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/gradients/dropout/dropout/div_grad/BroadcastGradientArgs"
  op: "BroadcastGradientArgs"
  input: "train/gradients/dropout/dropout/div_grad/Shape"
  input: "train/gradients/dropout/dropout/div_grad/Shape_1"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/gradients/dropout/dropout/div_grad/RealDiv"
  op: "RealDiv"
  input: "train/gradients/dropout/dropout/mul_grad/tuple/control_dependency"
  input: "dropout/Placeholder"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "train/gradients/dropout/dropout/div_grad/Sum"
  op: "Sum"
  input: "train/gradients/dropout/dropout/div_grad/RealDiv"
  input: "train/gradients/dropout/dropout/div_grad/BroadcastGradientArgs"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "train/gradients/dropout/dropout/div_grad/Reshape"
  op: "Reshape"
  input: "train/gradients/dropout/dropout/div_grad/Sum"
  input: "train/gradients/dropout/dropout/div_grad/Shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/gradients/dropout/dropout/div_grad/Neg"
  op: "Neg"
  input: "layer1/activation"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "train/gradients/dropout/dropout/div_grad/RealDiv_1"
  op: "RealDiv"
  input: "train/gradients/dropout/dropout/div_grad/Neg"
  input: "dropout/Placeholder"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "train/gradients/dropout/dropout/div_grad/RealDiv_2"
  op: "RealDiv"
  input: "train/gradients/dropout/dropout/div_grad/RealDiv_1"
  input: "dropout/Placeholder"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "train/gradients/dropout/dropout/div_grad/mul"
  op: "Mul"
  input: "train/gradients/dropout/dropout/mul_grad/tuple/control_dependency"
  input: "train/gradients/dropout/dropout/div_grad/RealDiv_2"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "train/gradients/dropout/dropout/div_grad/Sum_1"
  op: "Sum"
  input: "train/gradients/dropout/dropout/div_grad/mul"
  input: "train/gradients/dropout/dropout/div_grad/BroadcastGradientArgs:1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "train/gradients/dropout/dropout/div_grad/Reshape_1"
  op: "Reshape"
  input: "train/gradients/dropout/dropout/div_grad/Sum_1"
  input: "train/gradients/dropout/dropout/div_grad/Shape_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/gradients/dropout/dropout/div_grad/tuple/group_deps"
  op: "NoOp"
  input: "^train/gradients/dropout/dropout/div_grad/Reshape"
  input: "^train/gradients/dropout/dropout/div_grad/Reshape_1"
}
node {
  name: "train/gradients/dropout/dropout/div_grad/tuple/control_dependency"
  op: "Identity"
  input: "train/gradients/dropout/dropout/div_grad/Reshape"
  input: "^train/gradients/dropout/dropout/div_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/gradients/dropout/dropout/div_grad/Reshape"
      }
    }
  }
}
node {
  name: "train/gradients/dropout/dropout/div_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "train/gradients/dropout/dropout/div_grad/Reshape_1"
  input: "^train/gradients/dropout/dropout/div_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/gradients/dropout/dropout/div_grad/Reshape_1"
      }
    }
  }
}
node {
  name: "train/gradients/layer1/activation_grad/ReluGrad"
  op: "ReluGrad"
  input: "train/gradients/dropout/dropout/div_grad/tuple/control_dependency"
  input: "layer1/activation"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "train/gradients/layer1/linear_compute/add_grad/Shape"
  op: "Shape"
  input: "layer1/linear_compute/MatMul"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/gradients/layer1/linear_compute/add_grad/Shape_1"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 500
      }
    }
  }
}
node {
  name: "train/gradients/layer1/linear_compute/add_grad/BroadcastGradientArgs"
  op: "BroadcastGradientArgs"
  input: "train/gradients/layer1/linear_compute/add_grad/Shape"
  input: "train/gradients/layer1/linear_compute/add_grad/Shape_1"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/gradients/layer1/linear_compute/add_grad/Sum"
  op: "Sum"
  input: "train/gradients/layer1/activation_grad/ReluGrad"
  input: "train/gradients/layer1/linear_compute/add_grad/BroadcastGradientArgs"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "train/gradients/layer1/linear_compute/add_grad/Reshape"
  op: "Reshape"
  input: "train/gradients/layer1/linear_compute/add_grad/Sum"
  input: "train/gradients/layer1/linear_compute/add_grad/Shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/gradients/layer1/linear_compute/add_grad/Sum_1"
  op: "Sum"
  input: "train/gradients/layer1/activation_grad/ReluGrad"
  input: "train/gradients/layer1/linear_compute/add_grad/BroadcastGradientArgs:1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "train/gradients/layer1/linear_compute/add_grad/Reshape_1"
  op: "Reshape"
  input: "train/gradients/layer1/linear_compute/add_grad/Sum_1"
  input: "train/gradients/layer1/linear_compute/add_grad/Shape_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "train/gradients/layer1/linear_compute/add_grad/tuple/group_deps"
  op: "NoOp"
  input: "^train/gradients/layer1/linear_compute/add_grad/Reshape"
  input: "^train/gradients/layer1/linear_compute/add_grad/Reshape_1"
}
node {
  name: "train/gradients/layer1/linear_compute/add_grad/tuple/control_dependency"
  op: "Identity"
  input: "train/gradients/layer1/linear_compute/add_grad/Reshape"
  input: "^train/gradients/layer1/linear_compute/add_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/gradients/layer1/linear_compute/add_grad/Reshape"
      }
    }
  }
}
node {
  name: "train/gradients/layer1/linear_compute/add_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "train/gradients/layer1/linear_compute/add_grad/Reshape_1"
  input: "^train/gradients/layer1/linear_compute/add_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/gradients/layer1/linear_compute/add_grad/Reshape_1"
      }
    }
  }
}
node {
  name: "train/gradients/layer1/linear_compute/MatMul_grad/MatMul"
  op: "MatMul"
  input: "train/gradients/layer1/linear_compute/add_grad/tuple/control_dependency"
  input: "layer1/weights/Variable/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "transpose_a"
    value {
      b: false
    }
  }
  attr {
    key: "transpose_b"
    value {
      b: true
    }
  }
}
node {
  name: "train/gradients/layer1/linear_compute/MatMul_grad/MatMul_1"
  op: "MatMul"
  input: "input/x-input"
  input: "train/gradients/layer1/linear_compute/add_grad/tuple/control_dependency"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "transpose_a"
    value {
      b: true
    }
  }
  attr {
    key: "transpose_b"
    value {
      b: false
    }
  }
}
node {
  name: "train/gradients/layer1/linear_compute/MatMul_grad/tuple/group_deps"
  op: "NoOp"
  input: "^train/gradients/layer1/linear_compute/MatMul_grad/MatMul"
  input: "^train/gradients/layer1/linear_compute/MatMul_grad/MatMul_1"
}
node {
  name: "train/gradients/layer1/linear_compute/MatMul_grad/tuple/control_dependency"
  op: "Identity"
  input: "train/gradients/layer1/linear_compute/MatMul_grad/MatMul"
  input: "^train/gradients/layer1/linear_compute/MatMul_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/gradients/layer1/linear_compute/MatMul_grad/MatMul"
      }
    }
  }
}
node {
  name: "train/gradients/layer1/linear_compute/MatMul_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "train/gradients/layer1/linear_compute/MatMul_grad/MatMul_1"
  input: "^train/gradients/layer1/linear_compute/MatMul_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@train/gradients/layer1/linear_compute/MatMul_grad/MatMul_1"
      }
    }
  }
}
node {
  name: "train/beta1_power/initial_value"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@layer1/weights/Variable"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.8999999761581421
      }
    }
  }
}
node {
  name: "train/beta1_power"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@layer1/weights/Variable"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "train/beta1_power/Assign"
  op: "Assign"
  input: "train/beta1_power"
  input: "train/beta1_power/initial_value"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@layer1/weights/Variable"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "train/beta1_power/read"
  op: "Identity"
  input: "train/beta1_power"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@layer1/weights/Variable"
      }
    }
  }
}
node {
  name: "train/beta2_power/initial_value"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@layer1/weights/Variable"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.9990000128746033
      }
    }
  }
}
node {
  name: "train/beta2_power"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@layer1/weights/Variable"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "train/beta2_power/Assign"
  op: "Assign"
  input: "train/beta2_power"
  input: "train/beta2_power/initial_value"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@layer1/weights/Variable"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "train/beta2_power/read"
  op: "Identity"
  input: "train/beta2_power"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@layer1/weights/Variable"
      }
    }
  }
}
node {
  name: "layer1/weights/Variable/Adam/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@layer1/weights/Variable"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 784
          }
          dim {
            size: 500
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "layer1/weights/Variable/Adam"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@layer1/weights/Variable"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 784
        }
        dim {
          size: 500
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "layer1/weights/Variable/Adam/Assign"
  op: "Assign"
  input: "layer1/weights/Variable/Adam"
  input: "layer1/weights/Variable/Adam/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@layer1/weights/Variable"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "layer1/weights/Variable/Adam/read"
  op: "Identity"
  input: "layer1/weights/Variable/Adam"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@layer1/weights/Variable"
      }
    }
  }
}
node {
  name: "layer1/weights/Variable/Adam_1/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@layer1/weights/Variable"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 784
          }
          dim {
            size: 500
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "layer1/weights/Variable/Adam_1"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@layer1/weights/Variable"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 784
        }
        dim {
          size: 500
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "layer1/weights/Variable/Adam_1/Assign"
  op: "Assign"
  input: "layer1/weights/Variable/Adam_1"
  input: "layer1/weights/Variable/Adam_1/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@layer1/weights/Variable"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "layer1/weights/Variable/Adam_1/read"
  op: "Identity"
  input: "layer1/weights/Variable/Adam_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@layer1/weights/Variable"
      }
    }
  }
}
node {
  name: "layer1/biases/Variable/Adam/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@layer1/biases/Variable"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 500
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "layer1/biases/Variable/Adam"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@layer1/biases/Variable"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 500
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "layer1/biases/Variable/Adam/Assign"
  op: "Assign"
  input: "layer1/biases/Variable/Adam"
  input: "layer1/biases/Variable/Adam/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@layer1/biases/Variable"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "layer1/biases/Variable/Adam/read"
  op: "Identity"
  input: "layer1/biases/Variable/Adam"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@layer1/biases/Variable"
      }
    }
  }
}
node {
  name: "layer1/biases/Variable/Adam_1/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@layer1/biases/Variable"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 500
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "layer1/biases/Variable/Adam_1"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@layer1/biases/Variable"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 500
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "layer1/biases/Variable/Adam_1/Assign"
  op: "Assign"
  input: "layer1/biases/Variable/Adam_1"
  input: "layer1/biases/Variable/Adam_1/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@layer1/biases/Variable"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "layer1/biases/Variable/Adam_1/read"
  op: "Identity"
  input: "layer1/biases/Variable/Adam_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@layer1/biases/Variable"
      }
    }
  }
}
node {
  name: "layer2/weights/Variable/Adam/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@layer2/weights/Variable"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 500
          }
          dim {
            size: 10
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "layer2/weights/Variable/Adam"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@layer2/weights/Variable"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 500
        }
        dim {
          size: 10
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "layer2/weights/Variable/Adam/Assign"
  op: "Assign"
  input: "layer2/weights/Variable/Adam"
  input: "layer2/weights/Variable/Adam/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@layer2/weights/Variable"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "layer2/weights/Variable/Adam/read"
  op: "Identity"
  input: "layer2/weights/Variable/Adam"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@layer2/weights/Variable"
      }
    }
  }
}
node {
  name: "layer2/weights/Variable/Adam_1/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@layer2/weights/Variable"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 500
          }
          dim {
            size: 10
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "layer2/weights/Variable/Adam_1"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@layer2/weights/Variable"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 500
        }
        dim {
          size: 10
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "layer2/weights/Variable/Adam_1/Assign"
  op: "Assign"
  input: "layer2/weights/Variable/Adam_1"
  input: "layer2/weights/Variable/Adam_1/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@layer2/weights/Variable"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "layer2/weights/Variable/Adam_1/read"
  op: "Identity"
  input: "layer2/weights/Variable/Adam_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@layer2/weights/Variable"
      }
    }
  }
}
node {
  name: "layer2/biases/Variable/Adam/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@layer2/biases/Variable"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 10
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "layer2/biases/Variable/Adam"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@layer2/biases/Variable"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 10
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "layer2/biases/Variable/Adam/Assign"
  op: "Assign"
  input: "layer2/biases/Variable/Adam"
  input: "layer2/biases/Variable/Adam/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@layer2/biases/Variable"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "layer2/biases/Variable/Adam/read"
  op: "Identity"
  input: "layer2/biases/Variable/Adam"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@layer2/biases/Variable"
      }
    }
  }
}
node {
  name: "layer2/biases/Variable/Adam_1/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@layer2/biases/Variable"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 10
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "layer2/biases/Variable/Adam_1"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@layer2/biases/Variable"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 10
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "layer2/biases/Variable/Adam_1/Assign"
  op: "Assign"
  input: "layer2/biases/Variable/Adam_1"
  input: "layer2/biases/Variable/Adam_1/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@layer2/biases/Variable"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "layer2/biases/Variable/Adam_1/read"
  op: "Identity"
  input: "layer2/biases/Variable/Adam_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@layer2/biases/Variable"
      }
    }
  }
}
node {
  name: "train/Adam/learning_rate"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 9.999999747378752e-05
      }
    }
  }
}
node {
  name: "train/Adam/beta1"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.8999999761581421
      }
    }
  }
}
node {
  name: "train/Adam/beta2"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.9990000128746033
      }
    }
  }
}
node {
  name: "train/Adam/epsilon"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 9.99999993922529e-09
      }
    }
  }
}
node {
  name: "train/Adam/update_layer1/weights/Variable/ApplyAdam"
  op: "ApplyAdam"
  input: "layer1/weights/Variable"
  input: "layer1/weights/Variable/Adam"
  input: "layer1/weights/Variable/Adam_1"
  input: "train/beta1_power/read"
  input: "train/beta2_power/read"
  input: "train/Adam/learning_rate"
  input: "train/Adam/beta1"
  input: "train/Adam/beta2"
  input: "train/Adam/epsilon"
  input: "train/gradients/layer1/linear_compute/MatMul_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@layer1/weights/Variable"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: false
    }
  }
  attr {
    key: "use_nesterov"
    value {
      b: false
    }
  }
}
node {
  name: "train/Adam/update_layer1/biases/Variable/ApplyAdam"
  op: "ApplyAdam"
  input: "layer1/biases/Variable"
  input: "layer1/biases/Variable/Adam"
  input: "layer1/biases/Variable/Adam_1"
  input: "train/beta1_power/read"
  input: "train/beta2_power/read"
  input: "train/Adam/learning_rate"
  input: "train/Adam/beta1"
  input: "train/Adam/beta2"
  input: "train/Adam/epsilon"
  input: "train/gradients/layer1/linear_compute/add_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@layer1/biases/Variable"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: false
    }
  }
  attr {
    key: "use_nesterov"
    value {
      b: false
    }
  }
}
node {
  name: "train/Adam/update_layer2/weights/Variable/ApplyAdam"
  op: "ApplyAdam"
  input: "layer2/weights/Variable"
  input: "layer2/weights/Variable/Adam"
  input: "layer2/weights/Variable/Adam_1"
  input: "train/beta1_power/read"
  input: "train/beta2_power/read"
  input: "train/Adam/learning_rate"
  input: "train/Adam/beta1"
  input: "train/Adam/beta2"
  input: "train/Adam/epsilon"
  input: "train/gradients/layer2/linear_compute/MatMul_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@layer2/weights/Variable"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: false
    }
  }
  attr {
    key: "use_nesterov"
    value {
      b: false
    }
  }
}
node {
  name: "train/Adam/update_layer2/biases/Variable/ApplyAdam"
  op: "ApplyAdam"
  input: "layer2/biases/Variable"
  input: "layer2/biases/Variable/Adam"
  input: "layer2/biases/Variable/Adam_1"
  input: "train/beta1_power/read"
  input: "train/beta2_power/read"
  input: "train/Adam/learning_rate"
  input: "train/Adam/beta1"
  input: "train/Adam/beta2"
  input: "train/Adam/epsilon"
  input: "train/gradients/layer2/linear_compute/add_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@layer2/biases/Variable"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: false
    }
  }
  attr {
    key: "use_nesterov"
    value {
      b: false
    }
  }
}
node {
  name: "train/Adam/mul"
  op: "Mul"
  input: "train/beta1_power/read"
  input: "train/Adam/beta1"
  input: "^train/Adam/update_layer1/weights/Variable/ApplyAdam"
  input: "^train/Adam/update_layer1/biases/Variable/ApplyAdam"
  input: "^train/Adam/update_layer2/weights/Variable/ApplyAdam"
  input: "^train/Adam/update_layer2/biases/Variable/ApplyAdam"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@layer1/weights/Variable"
      }
    }
  }
}
node {
  name: "train/Adam/Assign"
  op: "Assign"
  input: "train/beta1_power"
  input: "train/Adam/mul"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@layer1/weights/Variable"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: false
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "train/Adam/mul_1"
  op: "Mul"
  input: "train/beta2_power/read"
  input: "train/Adam/beta2"
  input: "^train/Adam/update_layer1/weights/Variable/ApplyAdam"
  input: "^train/Adam/update_layer1/biases/Variable/ApplyAdam"
  input: "^train/Adam/update_layer2/weights/Variable/ApplyAdam"
  input: "^train/Adam/update_layer2/biases/Variable/ApplyAdam"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@layer1/weights/Variable"
      }
    }
  }
}
node {
  name: "train/Adam/Assign_1"
  op: "Assign"
  input: "train/beta2_power"
  input: "train/Adam/mul_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@layer1/weights/Variable"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: false
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "train/Adam"
  op: "NoOp"
  input: "^train/Adam/update_layer1/weights/Variable/ApplyAdam"
  input: "^train/Adam/update_layer1/biases/Variable/ApplyAdam"
  input: "^train/Adam/update_layer2/weights/Variable/ApplyAdam"
  input: "^train/Adam/update_layer2/biases/Variable/ApplyAdam"
  input: "^train/Adam/Assign"
  input: "^train/Adam/Assign_1"
}
node {
  name: "accuracy/correct_prediction/ArgMax/dimension"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "accuracy/correct_prediction/ArgMax"
  op: "ArgMax"
  input: "layer2/activation"
  input: "accuracy/correct_prediction/ArgMax/dimension"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "accuracy/correct_prediction/ArgMax_1/dimension"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "accuracy/correct_prediction/ArgMax_1"
  op: "ArgMax"
  input: "input/y-input"
  input: "accuracy/correct_prediction/ArgMax_1/dimension"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "accuracy/correct_prediction/Equal"
  op: "Equal"
  input: "accuracy/correct_prediction/ArgMax"
  input: "accuracy/correct_prediction/ArgMax_1"
  attr {
    key: "T"
    value {
      type: DT_INT64
    }
  }
}
node {
  name: "accuracy/accuracy/Cast"
  op: "Cast"
  input: "accuracy/correct_prediction/Equal"
  attr {
    key: "DstT"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "SrcT"
    value {
      type: DT_BOOL
    }
  }
}
node {
  name: "accuracy/accuracy/Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 0
      }
    }
  }
}
node {
  name: "accuracy/accuracy/Mean"
  op: "Mean"
  input: "accuracy/accuracy/Cast"
  input: "accuracy/accuracy/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "accuracy_1/tags"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
        }
        string_val: "accuracy_1"
      }
    }
  }
}
node {
  name: "accuracy_1"
  op: "ScalarSummary"
  input: "accuracy_1/tags"
  input: "accuracy/accuracy/Mean"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "Merge/MergeSummary"
  op: "MergeSummary"
  input: "input_reshape/input"
  input: "layer1/weights/summaries/mean"
  input: "layer1/weights/summaries/stddev_1"
  input: "layer1/weights/summaries/max"
  input: "layer1/weights/summaries/min"
  input: "layer1/biases/summaries/mean"
  input: "layer1/biases/summaries/stddev_1"
  input: "layer1/biases/summaries/max"
  input: "layer1/biases/summaries/min"
  input: "dropout/dropout_keep_probability"
  input: "layer2/weights/summaries/mean"
  input: "layer2/weights/summaries/stddev_1"
  input: "layer2/weights/summaries/max"
  input: "layer2/weights/summaries/min"
  input: "layer2/biases/summaries/mean"
  input: "layer2/biases/summaries/stddev_1"
  input: "layer2/biases/summaries/max"
  input: "layer2/biases/summaries/min"
  input: "loss/loss"
  input: "accuracy_1"
  attr {
    key: "N"
    value {
      i: 20
    }
  }
}
node {
  name: "init"
  op: "NoOp"
  input: "^layer1/weights/Variable/Assign"
  input: "^layer1/biases/Variable/Assign"
  input: "^layer2/weights/Variable/Assign"
  input: "^layer2/biases/Variable/Assign"
  input: "^train/beta1_power/Assign"
  input: "^train/beta2_power/Assign"
  input: "^layer1/weights/Variable/Adam/Assign"
  input: "^layer1/weights/Variable/Adam_1/Assign"
  input: "^layer1/biases/Variable/Adam/Assign"
  input: "^layer1/biases/Variable/Adam_1/Assign"
  input: "^layer2/weights/Variable/Adam/Assign"
  input: "^layer2/weights/Variable/Adam_1/Assign"
  input: "^layer2/biases/Variable/Adam/Assign"
  input: "^layer2/biases/Variable/Adam_1/Assign"
}
node {
  name: "save/Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
        }
        string_val: "model"
      }
    }
  }
}
node {
  name: "save/SaveV2/tensor_names"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 14
          }
        }
        string_val: "layer1/biases/Variable"
        string_val: "layer1/biases/Variable/Adam"
        string_val: "layer1/biases/Variable/Adam_1"
        string_val: "layer1/weights/Variable"
        string_val: "layer1/weights/Variable/Adam"
        string_val: "layer1/weights/Variable/Adam_1"
        string_val: "layer2/biases/Variable"
        string_val: "layer2/biases/Variable/Adam"
        string_val: "layer2/biases/Variable/Adam_1"
        string_val: "layer2/weights/Variable"
        string_val: "layer2/weights/Variable/Adam"
        string_val: "layer2/weights/Variable/Adam_1"
        string_val: "train/beta1_power"
        string_val: "train/beta2_power"
      }
    }
  }
}
node {
  name: "save/SaveV2/shape_and_slices"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 14
          }
        }
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
      }
    }
  }
}
node {
  name: "save/SaveV2"
  op: "SaveV2"
  input: "save/Const"
  input: "save/SaveV2/tensor_names"
  input: "save/SaveV2/shape_and_slices"
  input: "layer1/biases/Variable"
  input: "layer1/biases/Variable/Adam"
  input: "layer1/biases/Variable/Adam_1"
  input: "layer1/weights/Variable"
  input: "layer1/weights/Variable/Adam"
  input: "layer1/weights/Variable/Adam_1"
  input: "layer2/biases/Variable"
  input: "layer2/biases/Variable/Adam"
  input: "layer2/biases/Variable/Adam_1"
  input: "layer2/weights/Variable"
  input: "layer2/weights/Variable/Adam"
  input: "layer2/weights/Variable/Adam_1"
  input: "train/beta1_power"
  input: "train/beta2_power"
  attr {
    key: "dtypes"
    value {
      list {
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
      }
    }
  }
}
node {
  name: "save/control_dependency"
  op: "Identity"
  input: "save/Const"
  input: "^save/SaveV2"
  attr {
    key: "T"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@save/Const"
      }
    }
  }
}
node {
  name: "save/RestoreV2/tensor_names"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 1
          }
        }
        string_val: "layer1/biases/Variable"
      }
    }
  }
}
node {
  name: "save/RestoreV2/shape_and_slices"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 1
          }
        }
        string_val: ""
      }
    }
  }
}
node {
  name: "save/RestoreV2"
  op: "RestoreV2"
  input: "save/Const"
  input: "save/RestoreV2/tensor_names"
  input: "save/RestoreV2/shape_and_slices"
  attr {
    key: "dtypes"
    value {
      list {
        type: DT_FLOAT
      }
    }
  }
}
node {
  name: "save/Assign"
  op: "Assign"
  input: "layer1/biases/Variable"
  input: "save/RestoreV2"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@layer1/biases/Variable"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/RestoreV2_1/tensor_names"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 1
          }
        }
        string_val: "layer1/biases/Variable/Adam"
      }
    }
  }
}
node {
  name: "save/RestoreV2_1/shape_and_slices"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 1
          }
        }
        string_val: ""
      }
    }
  }
}
node {
  name: "save/RestoreV2_1"
  op: "RestoreV2"
  input: "save/Const"
  input: "save/RestoreV2_1/tensor_names"
  input: "save/RestoreV2_1/shape_and_slices"
  attr {
    key: "dtypes"
    value {
      list {
        type: DT_FLOAT
      }
    }
  }
}
node {
  name: "save/Assign_1"
  op: "Assign"
  input: "layer1/biases/Variable/Adam"
  input: "save/RestoreV2_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@layer1/biases/Variable"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/RestoreV2_2/tensor_names"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 1
          }
        }
        string_val: "layer1/biases/Variable/Adam_1"
      }
    }
  }
}
node {
  name: "save/RestoreV2_2/shape_and_slices"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 1
          }
        }
        string_val: ""
      }
    }
  }
}
node {
  name: "save/RestoreV2_2"
  op: "RestoreV2"
  input: "save/Const"
  input: "save/RestoreV2_2/tensor_names"
  input: "save/RestoreV2_2/shape_and_slices"
  attr {
    key: "dtypes"
    value {
      list {
        type: DT_FLOAT
      }
    }
  }
}
node {
  name: "save/Assign_2"
  op: "Assign"
  input: "layer1/biases/Variable/Adam_1"
  input: "save/RestoreV2_2"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@layer1/biases/Variable"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/RestoreV2_3/tensor_names"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 1
          }
        }
        string_val: "layer1/weights/Variable"
      }
    }
  }
}
node {
  name: "save/RestoreV2_3/shape_and_slices"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 1
          }
        }
        string_val: ""
      }
    }
  }
}
node {
  name: "save/RestoreV2_3"
  op: "RestoreV2"
  input: "save/Const"
  input: "save/RestoreV2_3/tensor_names"
  input: "save/RestoreV2_3/shape_and_slices"
  attr {
    key: "dtypes"
    value {
      list {
        type: DT_FLOAT
      }
    }
  }
}
node {
  name: "save/Assign_3"
  op: "Assign"
  input: "layer1/weights/Variable"
  input: "save/RestoreV2_3"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@layer1/weights/Variable"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/RestoreV2_4/tensor_names"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 1
          }
        }
        string_val: "layer1/weights/Variable/Adam"
      }
    }
  }
}
node {
  name: "save/RestoreV2_4/shape_and_slices"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 1
          }
        }
        string_val: ""
      }
    }
  }
}
node {
  name: "save/RestoreV2_4"
  op: "RestoreV2"
  input: "save/Const"
  input: "save/RestoreV2_4/tensor_names"
  input: "save/RestoreV2_4/shape_and_slices"
  attr {
    key: "dtypes"
    value {
      list {
        type: DT_FLOAT
      }
    }
  }
}
node {
  name: "save/Assign_4"
  op: "Assign"
  input: "layer1/weights/Variable/Adam"
  input: "save/RestoreV2_4"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@layer1/weights/Variable"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/RestoreV2_5/tensor_names"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 1
          }
        }
        string_val: "layer1/weights/Variable/Adam_1"
      }
    }
  }
}
node {
  name: "save/RestoreV2_5/shape_and_slices"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 1
          }
        }
        string_val: ""
      }
    }
  }
}
node {
  name: "save/RestoreV2_5"
  op: "RestoreV2"
  input: "save/Const"
  input: "save/RestoreV2_5/tensor_names"
  input: "save/RestoreV2_5/shape_and_slices"
  attr {
    key: "dtypes"
    value {
      list {
        type: DT_FLOAT
      }
    }
  }
}
node {
  name: "save/Assign_5"
  op: "Assign"
  input: "layer1/weights/Variable/Adam_1"
  input: "save/RestoreV2_5"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@layer1/weights/Variable"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/RestoreV2_6/tensor_names"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 1
          }
        }
        string_val: "layer2/biases/Variable"
      }
    }
  }
}
node {
  name: "save/RestoreV2_6/shape_and_slices"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 1
          }
        }
        string_val: ""
      }
    }
  }
}
node {
  name: "save/RestoreV2_6"
  op: "RestoreV2"
  input: "save/Const"
  input: "save/RestoreV2_6/tensor_names"
  input: "save/RestoreV2_6/shape_and_slices"
  attr {
    key: "dtypes"
    value {
      list {
        type: DT_FLOAT
      }
    }
  }
}
node {
  name: "save/Assign_6"
  op: "Assign"
  input: "layer2/biases/Variable"
  input: "save/RestoreV2_6"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@layer2/biases/Variable"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/RestoreV2_7/tensor_names"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 1
          }
        }
        string_val: "layer2/biases/Variable/Adam"
      }
    }
  }
}
node {
  name: "save/RestoreV2_7/shape_and_slices"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 1
          }
        }
        string_val: ""
      }
    }
  }
}
node {
  name: "save/RestoreV2_7"
  op: "RestoreV2"
  input: "save/Const"
  input: "save/RestoreV2_7/tensor_names"
  input: "save/RestoreV2_7/shape_and_slices"
  attr {
    key: "dtypes"
    value {
      list {
        type: DT_FLOAT
      }
    }
  }
}
node {
  name: "save/Assign_7"
  op: "Assign"
  input: "layer2/biases/Variable/Adam"
  input: "save/RestoreV2_7"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@layer2/biases/Variable"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/RestoreV2_8/tensor_names"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 1
          }
        }
        string_val: "layer2/biases/Variable/Adam_1"
      }
    }
  }
}
node {
  name: "save/RestoreV2_8/shape_and_slices"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 1
          }
        }
        string_val: ""
      }
    }
  }
}
node {
  name: "save/RestoreV2_8"
  op: "RestoreV2"
  input: "save/Const"
  input: "save/RestoreV2_8/tensor_names"
  input: "save/RestoreV2_8/shape_and_slices"
  attr {
    key: "dtypes"
    value {
      list {
        type: DT_FLOAT
      }
    }
  }
}
node {
  name: "save/Assign_8"
  op: "Assign"
  input: "layer2/biases/Variable/Adam_1"
  input: "save/RestoreV2_8"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@layer2/biases/Variable"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/RestoreV2_9/tensor_names"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 1
          }
        }
        string_val: "layer2/weights/Variable"
      }
    }
  }
}
node {
  name: "save/RestoreV2_9/shape_and_slices"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 1
          }
        }
        string_val: ""
      }
    }
  }
}
node {
  name: "save/RestoreV2_9"
  op: "RestoreV2"
  input: "save/Const"
  input: "save/RestoreV2_9/tensor_names"
  input: "save/RestoreV2_9/shape_and_slices"
  attr {
    key: "dtypes"
    value {
      list {
        type: DT_FLOAT
      }
    }
  }
}
node {
  name: "save/Assign_9"
  op: "Assign"
  input: "layer2/weights/Variable"
  input: "save/RestoreV2_9"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@layer2/weights/Variable"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/RestoreV2_10/tensor_names"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 1
          }
        }
        string_val: "layer2/weights/Variable/Adam"
      }
    }
  }
}
node {
  name: "save/RestoreV2_10/shape_and_slices"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 1
          }
        }
        string_val: ""
      }
    }
  }
}
node {
  name: "save/RestoreV2_10"
  op: "RestoreV2"
  input: "save/Const"
  input: "save/RestoreV2_10/tensor_names"
  input: "save/RestoreV2_10/shape_and_slices"
  attr {
    key: "dtypes"
    value {
      list {
        type: DT_FLOAT
      }
    }
  }
}
node {
  name: "save/Assign_10"
  op: "Assign"
  input: "layer2/weights/Variable/Adam"
  input: "save/RestoreV2_10"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@layer2/weights/Variable"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/RestoreV2_11/tensor_names"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 1
          }
        }
        string_val: "layer2/weights/Variable/Adam_1"
      }
    }
  }
}
node {
  name: "save/RestoreV2_11/shape_and_slices"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 1
          }
        }
        string_val: ""
      }
    }
  }
}
node {
  name: "save/RestoreV2_11"
  op: "RestoreV2"
  input: "save/Const"
  input: "save/RestoreV2_11/tensor_names"
  input: "save/RestoreV2_11/shape_and_slices"
  attr {
    key: "dtypes"
    value {
      list {
        type: DT_FLOAT
      }
    }
  }
}
node {
  name: "save/Assign_11"
  op: "Assign"
  input: "layer2/weights/Variable/Adam_1"
  input: "save/RestoreV2_11"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@layer2/weights/Variable"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/RestoreV2_12/tensor_names"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 1
          }
        }
        string_val: "train/beta1_power"
      }
    }
  }
}
node {
  name: "save/RestoreV2_12/shape_and_slices"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 1
          }
        }
        string_val: ""
      }
    }
  }
}
node {
  name: "save/RestoreV2_12"
  op: "RestoreV2"
  input: "save/Const"
  input: "save/RestoreV2_12/tensor_names"
  input: "save/RestoreV2_12/shape_and_slices"
  attr {
    key: "dtypes"
    value {
      list {
        type: DT_FLOAT
      }
    }
  }
}
node {
  name: "save/Assign_12"
  op: "Assign"
  input: "train/beta1_power"
  input: "save/RestoreV2_12"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@layer1/weights/Variable"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/RestoreV2_13/tensor_names"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 1
          }
        }
        string_val: "train/beta2_power"
      }
    }
  }
}
node {
  name: "save/RestoreV2_13/shape_and_slices"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 1
          }
        }
        string_val: ""
      }
    }
  }
}
node {
  name: "save/RestoreV2_13"
  op: "RestoreV2"
  input: "save/Const"
  input: "save/RestoreV2_13/tensor_names"
  input: "save/RestoreV2_13/shape_and_slices"
  attr {
    key: "dtypes"
    value {
      list {
        type: DT_FLOAT
      }
    }
  }
}
node {
  name: "save/Assign_13"
  op: "Assign"
  input: "train/beta2_power"
  input: "save/RestoreV2_13"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@layer1/weights/Variable"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/restore_all"
  op: "NoOp"
  input: "^save/Assign"
  input: "^save/Assign_1"
  input: "^save/Assign_2"
  input: "^save/Assign_3"
  input: "^save/Assign_4"
  input: "^save/Assign_5"
  input: "^save/Assign_6"
  input: "^save/Assign_7"
  input: "^save/Assign_8"
  input: "^save/Assign_9"
  input: "^save/Assign_10"
  input: "^save/Assign_11"
  input: "^save/Assign_12"
  input: "^save/Assign_13"
}
node {
  name: "gradients/Shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
          }
        }
      }
    }
  }
}
node {
  name: "gradients/Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "gradients/Fill"
  op: "Fill"
  input: "gradients/Shape"
  input: "gradients/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "gradients/loss/total/Mean_grad/Reshape/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "gradients/loss/total/Mean_grad/Reshape"
  op: "Reshape"
  input: "gradients/Fill"
  input: "gradients/loss/total/Mean_grad/Reshape/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients/loss/total/Mean_grad/Shape"
  op: "Shape"
  input: "loss/Reshape_2"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients/loss/total/Mean_grad/Tile"
  op: "Tile"
  input: "gradients/loss/total/Mean_grad/Reshape"
  input: "gradients/loss/total/Mean_grad/Shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tmultiples"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients/loss/total/Mean_grad/Shape_1"
  op: "Shape"
  input: "loss/Reshape_2"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients/loss/total/Mean_grad/Shape_2"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
          }
        }
      }
    }
  }
}
node {
  name: "gradients/loss/total/Mean_grad/Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 0
      }
    }
  }
}
node {
  name: "gradients/loss/total/Mean_grad/Prod"
  op: "Prod"
  input: "gradients/loss/total/Mean_grad/Shape_1"
  input: "gradients/loss/total/Mean_grad/Const"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "gradients/loss/total/Mean_grad/Const_1"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 0
      }
    }
  }
}
node {
  name: "gradients/loss/total/Mean_grad/Prod_1"
  op: "Prod"
  input: "gradients/loss/total/Mean_grad/Shape_2"
  input: "gradients/loss/total/Mean_grad/Const_1"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "gradients/loss/total/Mean_grad/Maximum/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "gradients/loss/total/Mean_grad/Maximum"
  op: "Maximum"
  input: "gradients/loss/total/Mean_grad/Prod_1"
  input: "gradients/loss/total/Mean_grad/Maximum/y"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients/loss/total/Mean_grad/floordiv"
  op: "FloorDiv"
  input: "gradients/loss/total/Mean_grad/Prod"
  input: "gradients/loss/total/Mean_grad/Maximum"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients/loss/total/Mean_grad/Cast"
  op: "Cast"
  input: "gradients/loss/total/Mean_grad/floordiv"
  attr {
    key: "DstT"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "SrcT"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients/loss/total/Mean_grad/truediv"
  op: "RealDiv"
  input: "gradients/loss/total/Mean_grad/Tile"
  input: "gradients/loss/total/Mean_grad/Cast"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "gradients/loss/Reshape_2_grad/Shape"
  op: "Shape"
  input: "loss/SoftmaxCrossEntropyWithLogits"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients/loss/Reshape_2_grad/Reshape"
  op: "Reshape"
  input: "gradients/loss/total/Mean_grad/truediv"
  input: "gradients/loss/Reshape_2_grad/Shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients/zeros_like"
  op: "ZerosLike"
  input: "loss/SoftmaxCrossEntropyWithLogits:1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "gradients/loss/SoftmaxCrossEntropyWithLogits_grad/ExpandDims/dim"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: -1
      }
    }
  }
}
node {
  name: "gradients/loss/SoftmaxCrossEntropyWithLogits_grad/ExpandDims"
  op: "ExpandDims"
  input: "gradients/loss/Reshape_2_grad/Reshape"
  input: "gradients/loss/SoftmaxCrossEntropyWithLogits_grad/ExpandDims/dim"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tdim"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients/loss/SoftmaxCrossEntropyWithLogits_grad/mul"
  op: "Mul"
  input: "gradients/loss/SoftmaxCrossEntropyWithLogits_grad/ExpandDims"
  input: "loss/SoftmaxCrossEntropyWithLogits:1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "gradients/loss/Reshape_grad/Shape"
  op: "Shape"
  input: "layer2/activation"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients/loss/Reshape_grad/Reshape"
  op: "Reshape"
  input: "gradients/loss/SoftmaxCrossEntropyWithLogits_grad/mul"
  input: "gradients/loss/Reshape_grad/Shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients/layer2/linear_compute/add_grad/Shape"
  op: "Shape"
  input: "layer2/linear_compute/MatMul"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients/layer2/linear_compute/add_grad/Shape_1"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 10
      }
    }
  }
}
node {
  name: "gradients/layer2/linear_compute/add_grad/BroadcastGradientArgs"
  op: "BroadcastGradientArgs"
  input: "gradients/layer2/linear_compute/add_grad/Shape"
  input: "gradients/layer2/linear_compute/add_grad/Shape_1"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients/layer2/linear_compute/add_grad/Sum"
  op: "Sum"
  input: "gradients/loss/Reshape_grad/Reshape"
  input: "gradients/layer2/linear_compute/add_grad/BroadcastGradientArgs"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "gradients/layer2/linear_compute/add_grad/Reshape"
  op: "Reshape"
  input: "gradients/layer2/linear_compute/add_grad/Sum"
  input: "gradients/layer2/linear_compute/add_grad/Shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients/layer2/linear_compute/add_grad/Sum_1"
  op: "Sum"
  input: "gradients/loss/Reshape_grad/Reshape"
  input: "gradients/layer2/linear_compute/add_grad/BroadcastGradientArgs:1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "gradients/layer2/linear_compute/add_grad/Reshape_1"
  op: "Reshape"
  input: "gradients/layer2/linear_compute/add_grad/Sum_1"
  input: "gradients/layer2/linear_compute/add_grad/Shape_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients/layer2/linear_compute/add_grad/tuple/group_deps"
  op: "NoOp"
  input: "^gradients/layer2/linear_compute/add_grad/Reshape"
  input: "^gradients/layer2/linear_compute/add_grad/Reshape_1"
}
node {
  name: "gradients/layer2/linear_compute/add_grad/tuple/control_dependency"
  op: "Identity"
  input: "gradients/layer2/linear_compute/add_grad/Reshape"
  input: "^gradients/layer2/linear_compute/add_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@gradients/layer2/linear_compute/add_grad/Reshape"
      }
    }
  }
}
node {
  name: "gradients/layer2/linear_compute/add_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "gradients/layer2/linear_compute/add_grad/Reshape_1"
  input: "^gradients/layer2/linear_compute/add_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@gradients/layer2/linear_compute/add_grad/Reshape_1"
      }
    }
  }
}
node {
  name: "gradients/layer2/linear_compute/MatMul_grad/MatMul"
  op: "MatMul"
  input: "gradients/layer2/linear_compute/add_grad/tuple/control_dependency"
  input: "layer2/weights/Variable/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "transpose_a"
    value {
      b: false
    }
  }
  attr {
    key: "transpose_b"
    value {
      b: true
    }
  }
}
node {
  name: "gradients/layer2/linear_compute/MatMul_grad/MatMul_1"
  op: "MatMul"
  input: "dropout/dropout/mul"
  input: "gradients/layer2/linear_compute/add_grad/tuple/control_dependency"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "transpose_a"
    value {
      b: true
    }
  }
  attr {
    key: "transpose_b"
    value {
      b: false
    }
  }
}
node {
  name: "gradients/layer2/linear_compute/MatMul_grad/tuple/group_deps"
  op: "NoOp"
  input: "^gradients/layer2/linear_compute/MatMul_grad/MatMul"
  input: "^gradients/layer2/linear_compute/MatMul_grad/MatMul_1"
}
node {
  name: "gradients/layer2/linear_compute/MatMul_grad/tuple/control_dependency"
  op: "Identity"
  input: "gradients/layer2/linear_compute/MatMul_grad/MatMul"
  input: "^gradients/layer2/linear_compute/MatMul_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@gradients/layer2/linear_compute/MatMul_grad/MatMul"
      }
    }
  }
}
node {
  name: "gradients/layer2/linear_compute/MatMul_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "gradients/layer2/linear_compute/MatMul_grad/MatMul_1"
  input: "^gradients/layer2/linear_compute/MatMul_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@gradients/layer2/linear_compute/MatMul_grad/MatMul_1"
      }
    }
  }
}
node {
  name: "gradients/dropout/dropout/mul_grad/Shape"
  op: "Shape"
  input: "dropout/dropout/div"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients/dropout/dropout/mul_grad/Shape_1"
  op: "Shape"
  input: "dropout/dropout/Floor"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients/dropout/dropout/mul_grad/BroadcastGradientArgs"
  op: "BroadcastGradientArgs"
  input: "gradients/dropout/dropout/mul_grad/Shape"
  input: "gradients/dropout/dropout/mul_grad/Shape_1"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients/dropout/dropout/mul_grad/mul"
  op: "Mul"
  input: "gradients/layer2/linear_compute/MatMul_grad/tuple/control_dependency"
  input: "dropout/dropout/Floor"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "gradients/dropout/dropout/mul_grad/Sum"
  op: "Sum"
  input: "gradients/dropout/dropout/mul_grad/mul"
  input: "gradients/dropout/dropout/mul_grad/BroadcastGradientArgs"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "gradients/dropout/dropout/mul_grad/Reshape"
  op: "Reshape"
  input: "gradients/dropout/dropout/mul_grad/Sum"
  input: "gradients/dropout/dropout/mul_grad/Shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients/dropout/dropout/mul_grad/mul_1"
  op: "Mul"
  input: "dropout/dropout/div"
  input: "gradients/layer2/linear_compute/MatMul_grad/tuple/control_dependency"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "gradients/dropout/dropout/mul_grad/Sum_1"
  op: "Sum"
  input: "gradients/dropout/dropout/mul_grad/mul_1"
  input: "gradients/dropout/dropout/mul_grad/BroadcastGradientArgs:1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "gradients/dropout/dropout/mul_grad/Reshape_1"
  op: "Reshape"
  input: "gradients/dropout/dropout/mul_grad/Sum_1"
  input: "gradients/dropout/dropout/mul_grad/Shape_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients/dropout/dropout/mul_grad/tuple/group_deps"
  op: "NoOp"
  input: "^gradients/dropout/dropout/mul_grad/Reshape"
  input: "^gradients/dropout/dropout/mul_grad/Reshape_1"
}
node {
  name: "gradients/dropout/dropout/mul_grad/tuple/control_dependency"
  op: "Identity"
  input: "gradients/dropout/dropout/mul_grad/Reshape"
  input: "^gradients/dropout/dropout/mul_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@gradients/dropout/dropout/mul_grad/Reshape"
      }
    }
  }
}
node {
  name: "gradients/dropout/dropout/mul_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "gradients/dropout/dropout/mul_grad/Reshape_1"
  input: "^gradients/dropout/dropout/mul_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@gradients/dropout/dropout/mul_grad/Reshape_1"
      }
    }
  }
}
node {
  name: "gradients/dropout/dropout/div_grad/Shape"
  op: "Shape"
  input: "layer1/activation"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients/dropout/dropout/div_grad/Shape_1"
  op: "Shape"
  input: "dropout/Placeholder"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients/dropout/dropout/div_grad/BroadcastGradientArgs"
  op: "BroadcastGradientArgs"
  input: "gradients/dropout/dropout/div_grad/Shape"
  input: "gradients/dropout/dropout/div_grad/Shape_1"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients/dropout/dropout/div_grad/RealDiv"
  op: "RealDiv"
  input: "gradients/dropout/dropout/mul_grad/tuple/control_dependency"
  input: "dropout/Placeholder"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "gradients/dropout/dropout/div_grad/Sum"
  op: "Sum"
  input: "gradients/dropout/dropout/div_grad/RealDiv"
  input: "gradients/dropout/dropout/div_grad/BroadcastGradientArgs"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "gradients/dropout/dropout/div_grad/Reshape"
  op: "Reshape"
  input: "gradients/dropout/dropout/div_grad/Sum"
  input: "gradients/dropout/dropout/div_grad/Shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients/dropout/dropout/div_grad/Neg"
  op: "Neg"
  input: "layer1/activation"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "gradients/dropout/dropout/div_grad/RealDiv_1"
  op: "RealDiv"
  input: "gradients/dropout/dropout/div_grad/Neg"
  input: "dropout/Placeholder"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "gradients/dropout/dropout/div_grad/RealDiv_2"
  op: "RealDiv"
  input: "gradients/dropout/dropout/div_grad/RealDiv_1"
  input: "dropout/Placeholder"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "gradients/dropout/dropout/div_grad/mul"
  op: "Mul"
  input: "gradients/dropout/dropout/mul_grad/tuple/control_dependency"
  input: "gradients/dropout/dropout/div_grad/RealDiv_2"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "gradients/dropout/dropout/div_grad/Sum_1"
  op: "Sum"
  input: "gradients/dropout/dropout/div_grad/mul"
  input: "gradients/dropout/dropout/div_grad/BroadcastGradientArgs:1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "gradients/dropout/dropout/div_grad/Reshape_1"
  op: "Reshape"
  input: "gradients/dropout/dropout/div_grad/Sum_1"
  input: "gradients/dropout/dropout/div_grad/Shape_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients/dropout/dropout/div_grad/tuple/group_deps"
  op: "NoOp"
  input: "^gradients/dropout/dropout/div_grad/Reshape"
  input: "^gradients/dropout/dropout/div_grad/Reshape_1"
}
node {
  name: "gradients/dropout/dropout/div_grad/tuple/control_dependency"
  op: "Identity"
  input: "gradients/dropout/dropout/div_grad/Reshape"
  input: "^gradients/dropout/dropout/div_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@gradients/dropout/dropout/div_grad/Reshape"
      }
    }
  }
}
node {
  name: "gradients/dropout/dropout/div_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "gradients/dropout/dropout/div_grad/Reshape_1"
  input: "^gradients/dropout/dropout/div_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@gradients/dropout/dropout/div_grad/Reshape_1"
      }
    }
  }
}
node {
  name: "gradients/layer1/activation_grad/ReluGrad"
  op: "ReluGrad"
  input: "gradients/dropout/dropout/div_grad/tuple/control_dependency"
  input: "layer1/activation"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "gradients/layer1/linear_compute/add_grad/Shape"
  op: "Shape"
  input: "layer1/linear_compute/MatMul"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients/layer1/linear_compute/add_grad/Shape_1"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 500
      }
    }
  }
}
node {
  name: "gradients/layer1/linear_compute/add_grad/BroadcastGradientArgs"
  op: "BroadcastGradientArgs"
  input: "gradients/layer1/linear_compute/add_grad/Shape"
  input: "gradients/layer1/linear_compute/add_grad/Shape_1"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients/layer1/linear_compute/add_grad/Sum"
  op: "Sum"
  input: "gradients/layer1/activation_grad/ReluGrad"
  input: "gradients/layer1/linear_compute/add_grad/BroadcastGradientArgs"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "gradients/layer1/linear_compute/add_grad/Reshape"
  op: "Reshape"
  input: "gradients/layer1/linear_compute/add_grad/Sum"
  input: "gradients/layer1/linear_compute/add_grad/Shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients/layer1/linear_compute/add_grad/Sum_1"
  op: "Sum"
  input: "gradients/layer1/activation_grad/ReluGrad"
  input: "gradients/layer1/linear_compute/add_grad/BroadcastGradientArgs:1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "gradients/layer1/linear_compute/add_grad/Reshape_1"
  op: "Reshape"
  input: "gradients/layer1/linear_compute/add_grad/Sum_1"
  input: "gradients/layer1/linear_compute/add_grad/Shape_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients/layer1/linear_compute/add_grad/tuple/group_deps"
  op: "NoOp"
  input: "^gradients/layer1/linear_compute/add_grad/Reshape"
  input: "^gradients/layer1/linear_compute/add_grad/Reshape_1"
}
node {
  name: "gradients/layer1/linear_compute/add_grad/tuple/control_dependency"
  op: "Identity"
  input: "gradients/layer1/linear_compute/add_grad/Reshape"
  input: "^gradients/layer1/linear_compute/add_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@gradients/layer1/linear_compute/add_grad/Reshape"
      }
    }
  }
}
node {
  name: "gradients/layer1/linear_compute/add_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "gradients/layer1/linear_compute/add_grad/Reshape_1"
  input: "^gradients/layer1/linear_compute/add_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@gradients/layer1/linear_compute/add_grad/Reshape_1"
      }
    }
  }
}
node {
  name: "gradients/layer1/linear_compute/MatMul_grad/MatMul"
  op: "MatMul"
  input: "gradients/layer1/linear_compute/add_grad/tuple/control_dependency"
  input: "layer1/weights/Variable/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "transpose_a"
    value {
      b: false
    }
  }
  attr {
    key: "transpose_b"
    value {
      b: true
    }
  }
}
node {
  name: "gradients/layer1/linear_compute/MatMul_grad/MatMul_1"
  op: "MatMul"
  input: "input/x-input"
  input: "gradients/layer1/linear_compute/add_grad/tuple/control_dependency"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "transpose_a"
    value {
      b: true
    }
  }
  attr {
    key: "transpose_b"
    value {
      b: false
    }
  }
}
node {
  name: "gradients/layer1/linear_compute/MatMul_grad/tuple/group_deps"
  op: "NoOp"
  input: "^gradients/layer1/linear_compute/MatMul_grad/MatMul"
  input: "^gradients/layer1/linear_compute/MatMul_grad/MatMul_1"
}
node {
  name: "gradients/layer1/linear_compute/MatMul_grad/tuple/control_dependency"
  op: "Identity"
  input: "gradients/layer1/linear_compute/MatMul_grad/MatMul"
  input: "^gradients/layer1/linear_compute/MatMul_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@gradients/layer1/linear_compute/MatMul_grad/MatMul"
      }
    }
  }
}
node {
  name: "gradients/layer1/linear_compute/MatMul_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "gradients/layer1/linear_compute/MatMul_grad/MatMul_1"
  input: "^gradients/layer1/linear_compute/MatMul_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@gradients/layer1/linear_compute/MatMul_grad/MatMul_1"
      }
    }
  }
}
node {
  name: "beta1_power/initial_value"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@layer1/weights/Variable"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.8999999761581421
      }
    }
  }
}
node {
  name: "beta1_power"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@layer1/weights/Variable"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "beta1_power/Assign"
  op: "Assign"
  input: "beta1_power"
  input: "beta1_power/initial_value"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@layer1/weights/Variable"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "beta1_power/read"
  op: "Identity"
  input: "beta1_power"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@layer1/weights/Variable"
      }
    }
  }
}
node {
  name: "beta2_power/initial_value"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@layer1/weights/Variable"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.9990000128746033
      }
    }
  }
}
node {
  name: "beta2_power"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@layer1/weights/Variable"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "beta2_power/Assign"
  op: "Assign"
  input: "beta2_power"
  input: "beta2_power/initial_value"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@layer1/weights/Variable"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "beta2_power/read"
  op: "Identity"
  input: "beta2_power"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@layer1/weights/Variable"
      }
    }
  }
}
node {
  name: "layer1/weights/Variable/Adam_2/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@layer1/weights/Variable"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 784
          }
          dim {
            size: 500
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "layer1/weights/Variable/Adam_2"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@layer1/weights/Variable"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 784
        }
        dim {
          size: 500
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "layer1/weights/Variable/Adam_2/Assign"
  op: "Assign"
  input: "layer1/weights/Variable/Adam_2"
  input: "layer1/weights/Variable/Adam_2/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@layer1/weights/Variable"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "layer1/weights/Variable/Adam_2/read"
  op: "Identity"
  input: "layer1/weights/Variable/Adam_2"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@layer1/weights/Variable"
      }
    }
  }
}
node {
  name: "layer1/weights/Variable/Adam_3/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@layer1/weights/Variable"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 784
          }
          dim {
            size: 500
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "layer1/weights/Variable/Adam_3"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@layer1/weights/Variable"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 784
        }
        dim {
          size: 500
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "layer1/weights/Variable/Adam_3/Assign"
  op: "Assign"
  input: "layer1/weights/Variable/Adam_3"
  input: "layer1/weights/Variable/Adam_3/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@layer1/weights/Variable"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "layer1/weights/Variable/Adam_3/read"
  op: "Identity"
  input: "layer1/weights/Variable/Adam_3"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@layer1/weights/Variable"
      }
    }
  }
}
node {
  name: "layer1/biases/Variable/Adam_2/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@layer1/biases/Variable"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 500
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "layer1/biases/Variable/Adam_2"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@layer1/biases/Variable"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 500
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "layer1/biases/Variable/Adam_2/Assign"
  op: "Assign"
  input: "layer1/biases/Variable/Adam_2"
  input: "layer1/biases/Variable/Adam_2/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@layer1/biases/Variable"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "layer1/biases/Variable/Adam_2/read"
  op: "Identity"
  input: "layer1/biases/Variable/Adam_2"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@layer1/biases/Variable"
      }
    }
  }
}
node {
  name: "layer1/biases/Variable/Adam_3/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@layer1/biases/Variable"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 500
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "layer1/biases/Variable/Adam_3"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@layer1/biases/Variable"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 500
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "layer1/biases/Variable/Adam_3/Assign"
  op: "Assign"
  input: "layer1/biases/Variable/Adam_3"
  input: "layer1/biases/Variable/Adam_3/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@layer1/biases/Variable"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "layer1/biases/Variable/Adam_3/read"
  op: "Identity"
  input: "layer1/biases/Variable/Adam_3"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@layer1/biases/Variable"
      }
    }
  }
}
node {
  name: "layer2/weights/Variable/Adam_2/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@layer2/weights/Variable"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 500
          }
          dim {
            size: 10
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "layer2/weights/Variable/Adam_2"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@layer2/weights/Variable"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 500
        }
        dim {
          size: 10
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "layer2/weights/Variable/Adam_2/Assign"
  op: "Assign"
  input: "layer2/weights/Variable/Adam_2"
  input: "layer2/weights/Variable/Adam_2/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@layer2/weights/Variable"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "layer2/weights/Variable/Adam_2/read"
  op: "Identity"
  input: "layer2/weights/Variable/Adam_2"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@layer2/weights/Variable"
      }
    }
  }
}
node {
  name: "layer2/weights/Variable/Adam_3/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@layer2/weights/Variable"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 500
          }
          dim {
            size: 10
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "layer2/weights/Variable/Adam_3"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@layer2/weights/Variable"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 500
        }
        dim {
          size: 10
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "layer2/weights/Variable/Adam_3/Assign"
  op: "Assign"
  input: "layer2/weights/Variable/Adam_3"
  input: "layer2/weights/Variable/Adam_3/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@layer2/weights/Variable"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "layer2/weights/Variable/Adam_3/read"
  op: "Identity"
  input: "layer2/weights/Variable/Adam_3"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@layer2/weights/Variable"
      }
    }
  }
}
node {
  name: "layer2/biases/Variable/Adam_2/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@layer2/biases/Variable"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 10
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "layer2/biases/Variable/Adam_2"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@layer2/biases/Variable"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 10
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "layer2/biases/Variable/Adam_2/Assign"
  op: "Assign"
  input: "layer2/biases/Variable/Adam_2"
  input: "layer2/biases/Variable/Adam_2/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@layer2/biases/Variable"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "layer2/biases/Variable/Adam_2/read"
  op: "Identity"
  input: "layer2/biases/Variable/Adam_2"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@layer2/biases/Variable"
      }
    }
  }
}
node {
  name: "layer2/biases/Variable/Adam_3/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@layer2/biases/Variable"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 10
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "layer2/biases/Variable/Adam_3"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@layer2/biases/Variable"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 10
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "layer2/biases/Variable/Adam_3/Assign"
  op: "Assign"
  input: "layer2/biases/Variable/Adam_3"
  input: "layer2/biases/Variable/Adam_3/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@layer2/biases/Variable"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "layer2/biases/Variable/Adam_3/read"
  op: "Identity"
  input: "layer2/biases/Variable/Adam_3"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@layer2/biases/Variable"
      }
    }
  }
}
node {
  name: "Adam/learning_rate"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 9.999999747378752e-05
      }
    }
  }
}
node {
  name: "Adam/beta1"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.8999999761581421
      }
    }
  }
}
node {
  name: "Adam/beta2"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.9990000128746033
      }
    }
  }
}
node {
  name: "Adam/epsilon"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 9.99999993922529e-09
      }
    }
  }
}
node {
  name: "Adam/update_layer1/weights/Variable/ApplyAdam"
  op: "ApplyAdam"
  input: "layer1/weights/Variable"
  input: "layer1/weights/Variable/Adam_2"
  input: "layer1/weights/Variable/Adam_3"
  input: "beta1_power/read"
  input: "beta2_power/read"
  input: "Adam/learning_rate"
  input: "Adam/beta1"
  input: "Adam/beta2"
  input: "Adam/epsilon"
  input: "gradients/layer1/linear_compute/MatMul_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@layer1/weights/Variable"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: false
    }
  }
  attr {
    key: "use_nesterov"
    value {
      b: false
    }
  }
}
node {
  name: "Adam/update_layer1/biases/Variable/ApplyAdam"
  op: "ApplyAdam"
  input: "layer1/biases/Variable"
  input: "layer1/biases/Variable/Adam_2"
  input: "layer1/biases/Variable/Adam_3"
  input: "beta1_power/read"
  input: "beta2_power/read"
  input: "Adam/learning_rate"
  input: "Adam/beta1"
  input: "Adam/beta2"
  input: "Adam/epsilon"
  input: "gradients/layer1/linear_compute/add_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@layer1/biases/Variable"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: false
    }
  }
  attr {
    key: "use_nesterov"
    value {
      b: false
    }
  }
}
node {
  name: "Adam/update_layer2/weights/Variable/ApplyAdam"
  op: "ApplyAdam"
  input: "layer2/weights/Variable"
  input: "layer2/weights/Variable/Adam_2"
  input: "layer2/weights/Variable/Adam_3"
  input: "beta1_power/read"
  input: "beta2_power/read"
  input: "Adam/learning_rate"
  input: "Adam/beta1"
  input: "Adam/beta2"
  input: "Adam/epsilon"
  input: "gradients/layer2/linear_compute/MatMul_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@layer2/weights/Variable"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: false
    }
  }
  attr {
    key: "use_nesterov"
    value {
      b: false
    }
  }
}
node {
  name: "Adam/update_layer2/biases/Variable/ApplyAdam"
  op: "ApplyAdam"
  input: "layer2/biases/Variable"
  input: "layer2/biases/Variable/Adam_2"
  input: "layer2/biases/Variable/Adam_3"
  input: "beta1_power/read"
  input: "beta2_power/read"
  input: "Adam/learning_rate"
  input: "Adam/beta1"
  input: "Adam/beta2"
  input: "Adam/epsilon"
  input: "gradients/layer2/linear_compute/add_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@layer2/biases/Variable"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: false
    }
  }
  attr {
    key: "use_nesterov"
    value {
      b: false
    }
  }
}
node {
  name: "Adam/mul"
  op: "Mul"
  input: "beta1_power/read"
  input: "Adam/beta1"
  input: "^Adam/update_layer1/weights/Variable/ApplyAdam"
  input: "^Adam/update_layer1/biases/Variable/ApplyAdam"
  input: "^Adam/update_layer2/weights/Variable/ApplyAdam"
  input: "^Adam/update_layer2/biases/Variable/ApplyAdam"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@layer1/weights/Variable"
      }
    }
  }
}
node {
  name: "Adam/Assign"
  op: "Assign"
  input: "beta1_power"
  input: "Adam/mul"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@layer1/weights/Variable"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: false
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "Adam/mul_1"
  op: "Mul"
  input: "beta2_power/read"
  input: "Adam/beta2"
  input: "^Adam/update_layer1/weights/Variable/ApplyAdam"
  input: "^Adam/update_layer1/biases/Variable/ApplyAdam"
  input: "^Adam/update_layer2/weights/Variable/ApplyAdam"
  input: "^Adam/update_layer2/biases/Variable/ApplyAdam"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@layer1/weights/Variable"
      }
    }
  }
}
node {
  name: "Adam/Assign_1"
  op: "Assign"
  input: "beta2_power"
  input: "Adam/mul_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@layer1/weights/Variable"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: false
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "Adam"
  op: "NoOp"
  input: "^Adam/update_layer1/weights/Variable/ApplyAdam"
  input: "^Adam/update_layer1/biases/Variable/ApplyAdam"
  input: "^Adam/update_layer2/weights/Variable/ApplyAdam"
  input: "^Adam/update_layer2/biases/Variable/ApplyAdam"
  input: "^Adam/Assign"
  input: "^Adam/Assign_1"
}
node {
  name: "init_1"
  op: "NoOp"
  input: "^layer1/weights/Variable/Assign"
  input: "^layer1/biases/Variable/Assign"
  input: "^layer2/weights/Variable/Assign"
  input: "^layer2/biases/Variable/Assign"
  input: "^train/beta1_power/Assign"
  input: "^train/beta2_power/Assign"
  input: "^layer1/weights/Variable/Adam/Assign"
  input: "^layer1/weights/Variable/Adam_1/Assign"
  input: "^layer1/biases/Variable/Adam/Assign"
  input: "^layer1/biases/Variable/Adam_1/Assign"
  input: "^layer2/weights/Variable/Adam/Assign"
  input: "^layer2/weights/Variable/Adam_1/Assign"
  input: "^layer2/biases/Variable/Adam/Assign"
  input: "^layer2/biases/Variable/Adam_1/Assign"
  input: "^beta1_power/Assign"
  input: "^beta2_power/Assign"
  input: "^layer1/weights/Variable/Adam_2/Assign"
  input: "^layer1/weights/Variable/Adam_3/Assign"
  input: "^layer1/biases/Variable/Adam_2/Assign"
  input: "^layer1/biases/Variable/Adam_3/Assign"
  input: "^layer2/weights/Variable/Adam_2/Assign"
  input: "^layer2/weights/Variable/Adam_3/Assign"
  input: "^layer2/biases/Variable/Adam_2/Assign"
  input: "^layer2/biases/Variable/Adam_3/Assign"
}
versions {
  producer: 22
}
