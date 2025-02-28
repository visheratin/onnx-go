package ir

import (
	"bytes"
	"encoding/binary"
	"io"
	"math"

	"github.com/pkg/errors"

	"gorgonia.org/tensor"
)

// Tensor returns a Gorgonia compatible tensor
func (tx *TensorProto) Tensor() (tensor.Tensor, error) {
	if tx.Segment != nil {
		return nil, errors.Wrap(ErrNotYetImplemented, "This tensor is segmented")
	}
	// Get the data type
	if tx.DataType == int32(TensorProto_UNDEFINED) {
		return nil, errors.New("This tensor datatype is undefined")
	}
	dt, err := TensorProto_DataType(tx.DataType).Dtype()
	if err != nil {
		return nil, err
	}
	var size = make([]int, len(tx.Dims))
	for i := range tx.Dims {
		size[i] = int(tx.Dims[i])
	}
	opts := []tensor.ConsOpt{tensor.WithShape(size...), tensor.Of(dt)}
	var consopts []tensor.ConsOpt
	switch dt {
	case tensor.Bool:
		consopts, err = generateConsOptsFromBoolTensor(tx)
	case tensor.Float32:
		consopts, err = generateConsOptsFromFloat32Tensor(tx)
	case tensor.Float64:
		consopts, err = generateConsOptsFromFloat64Tensor(tx)
	case tensor.Int64:
		consopts, err = generateConsOptsFromInt64Tensor(tx)
	case tensor.Int32:
		consopts, err = generateConsOptsFromInt32Tensor(tx)
	default:
		err = errors.Wrapf(ErrNotYetImplemented, "Unknown type %v", dt)
	}
	if err != nil {
		return nil, err
	}
	opts = append(opts, consopts...)
	return tensor.New(opts...), nil
}

func generateConsOptsFromBoolTensor(tx *TensorProto) ([]tensor.ConsOpt, error) {
	switch {
	case tx.Int32Data != nil:
		backing := make([]bool, len(tx.Int32Data))
		for i := 0; i < len(tx.Int32Data); i++ {
			if tx.Int32Data[i] == 1 {
				backing[i] = true
			}
		}
		return []tensor.ConsOpt{tensor.WithBacking(backing)}, nil
	case tx.RawData != nil:
		buf := bytes.NewReader(tx.RawData)
		element := make([]byte, 8)
		var err error
		var backing []bool
		for {
			var n int
			n, err = buf.Read(element)
			if err != nil || n != 8 {
				break
			}
			if element[7] == 1 {
				backing = append(backing, true)
			} else {
				backing = append(backing, false)
			}
		}
		if err != io.EOF {
			return nil, errors.Wrapf(ErrCorruptedData, "%v", err)
		}
		return []tensor.ConsOpt{tensor.WithBacking(backing)}, nil
	default:
		return nil, errors.New("No data found")
	}
}

func generateConsOptsFromFloat32Tensor(tx *TensorProto) ([]tensor.ConsOpt, error) {
	switch {
	case tx.RawData != nil:
		if len(tx.RawData) == 0 {
			return []tensor.ConsOpt{tensor.WithBacking([]float32{})}, nil
		}
		buf := bytes.NewReader(tx.RawData)
		element := make([]byte, 4)
		var err error
		var backing []float32
		for {
			var n int
			n, err = buf.Read(element)
			if err != nil || n != 4 {
				break
			}
			uintElement := binary.LittleEndian.Uint32(element)
			backing = append(backing, math.Float32frombits(uintElement))
		}
		if err != io.EOF {
			return nil, errors.Wrapf(ErrCorruptedData, "%v", err)
		}
		return []tensor.ConsOpt{tensor.WithBacking(backing)}, nil
	case tx.FloatData != nil:
		return []tensor.ConsOpt{tensor.WithBacking(tx.FloatData)}, nil
	case tx.FloatData == nil && tx.RawData == nil:
		return []tensor.ConsOpt{tensor.WithBacking([]float32{})}, nil
	default:
		return nil, errors.New("No data found")
	}
}

func generateConsOptsFromFloat64Tensor(tx *TensorProto) ([]tensor.ConsOpt, error) {
	switch {
	case tx.DoubleData != nil:
		return []tensor.ConsOpt{tensor.WithBacking(tx.DoubleData)}, nil
	case tx.RawData != nil:
		if len(tx.RawData) == 0 {
			return []tensor.ConsOpt{tensor.WithBacking([]float64{})}, nil
		}
		buf := bytes.NewReader(tx.RawData)
		element := make([]byte, 8)
		var err error
		var backing []float64
		for {
			var n int
			n, err = buf.Read(element)
			if err != nil || n != 8 {
				break
			}
			uintElement := binary.LittleEndian.Uint64(element)
			backing = append(backing, math.Float64frombits(uintElement))
		}
		if err != io.EOF {
			return nil, errors.Wrapf(ErrCorruptedData, "%v", err)
		}
		return []tensor.ConsOpt{tensor.WithBacking(backing)}, nil
	default:
		return nil, errors.New("No data found")
	}
}

func generateConsOptsFromInt64Tensor(tx *TensorProto) ([]tensor.ConsOpt, error) {
	switch {
	case tx.RawData != nil:
		buf := bytes.NewReader(tx.RawData)
		element := make([]byte, 8)
		var err error
		var backing []int64
		for {
			var n int
			n, err = buf.Read(element)
			if err != nil || n != 8 {
				break
			}
			uintElement := binary.LittleEndian.Uint64(element)
			backing = append(backing, int64(uintElement))
		}

		if err != io.EOF {
			return nil, errors.Wrapf(ErrCorruptedData, "%v", err)
		}
		return []tensor.ConsOpt{tensor.WithBacking(backing)}, nil
	case tx.Int64Data != nil:
		return []tensor.ConsOpt{tensor.WithBacking(tx.Int64Data)}, nil
	default:
		return nil, errors.New("No data found")
	}
}

func generateConsOptsFromInt32Tensor(tx *TensorProto) ([]tensor.ConsOpt, error) {
	switch {
	case tx.RawData != nil:
		buf := bytes.NewReader(tx.RawData)
		element := make([]byte, 4)
		var err error
		var backing []int32
		for {
			var n int
			n, err = buf.Read(element)
			if err != nil || n != 4 {
				break
			}
			uintElement := binary.LittleEndian.Uint32(element)
			backing = append(backing, int32(uintElement))
		}
		if err != io.EOF {
			return nil, errors.Wrapf(ErrCorruptedData, "%v", err)
		}
		return []tensor.ConsOpt{tensor.WithBacking(backing)}, nil
	case tx.Int32Data != nil:
		return []tensor.ConsOpt{tensor.WithBacking(tx.Int32Data)}, nil
	default:
		return nil, errors.New("No data found")
	}
}
