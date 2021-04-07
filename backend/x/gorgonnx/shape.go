package gorgonnx

import (
	"errors"

	"github.com/owulveryck/onnx-go"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func init() {
	register("Shape", func() operator { return new(shape) })
}

type shape struct{}

func (*shape) apply(graph *Graph, nodes ...*Node) error {
	if len(nodes) != 1 {
		return errors.New("wrong number of input nodes")
	}
	children := getOrderedChildren(graph.g, nodes[0])
	err := checkCondition(children, 1)
	if err != nil {
		return err
	}
	s := []int(children[0].gorgoniaNode.Shape())
	v := make([]int64, len(s))
	for i := range s {
		v[i] = int64(s[i])
	}
	t := tensor.New(tensor.WithShape(len(s)), tensor.WithBacking(v))
	nodes[0].gorgoniaNode = gorgonia.NewTensor(graph.exprgraph, t.Dtype(), t.Dims(), gorgonia.WithValue(t))

	return nil
}

func (*shape) init(onnx.Operation) error {
	return nil
}
