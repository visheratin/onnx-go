package gorgonnx

import (
	"errors"
	"fmt"

	"github.com/owulveryck/onnx-go"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

type rs struct {
	start, end, step int
}

func (s rs) Start() int { return s.start }
func (s rs) End() int   { return s.end }
func (s rs) Step() int  { return s.step }

type slice struct {
}

func init() {
	register("Slice", newSlice)
}

func newSlice() operator {
	return &slice{}
}

func (s *slice) apply(g *Graph, ns ...*Node) error {
	n := ns[0]
	children := getOrderedChildren(g.g, n)
	err := checkForNil(children)
	if err != nil {
		return err
	}
	if len(children) < 3 {
		return errors.New("slice requires at least three inputs")
	}
	data := children[0].gorgoniaNode.Value().(*tensor.Dense)
	fmt.Println(data)
	startsNode := children[1].gorgoniaNode
	if startsNode.Dims() > 1 {
		return errors.New("starts node be a 1-D tensor")
	}
	startsT, ok := startsNode.Value().(*tensor.Dense)
	if !ok {
		return errors.New("starts vector must be a dense tensor")
	}
	startsLen := []int(startsT.Shape())[0]
	starts := make([]int64, startsLen)
	for i := 0; i < startsLen; i++ {
		v, err := startsT.At(0)
		if err != nil {
			return err
		}
		starts[i] = v.(int64)
	}
	endsNode := children[2].gorgoniaNode
	if endsNode.Dims() > 1 {
		return errors.New("ends node must be a 1-D tensor")
	}
	endsT, ok := endsNode.Value().(*tensor.Dense)
	if !ok {
		return errors.New("ends vector must be a dense tensor")
	}
	endsLen := []int(endsT.Shape())[0]
	if endsLen != startsLen {
		return errors.New("all slice tensors must be of the same size")
	}
	ends := make([]int64, endsLen)
	for i := 0; i < endsLen; i++ {
		v, err := endsT.At(0)
		if err != nil {
			return err
		}
		ends[i] = v.(int64)
	}
	var axes []int64
	if len(children) > 3 {
		axesNode := children[3].gorgoniaNode
		if axesNode.Dims() > 1 {
			return errors.New("axes node must be a 1-D tensor")
		}
		axesT, ok := axesNode.Value().(*tensor.Dense)
		if !ok {
			return errors.New("axes vector must be a dense tensor")
		}
		axesLen := []int(axesT.Shape())[0]
		if axesLen != startsLen {
			return errors.New("all slice tensors must be of the same size")
		}
		axes = make([]int64, axesLen)
		for i := 0; i < axesLen; i++ {
			v, err := axesT.At(0)
			if err != nil {
				return err
			}
			axes[i] = v.(int64)
		}
	} else {
		axes = make([]int64, startsLen)
		for i := 0; i < startsLen; i++ {
			axes[i] = int64(i)
		}
	}
	var steps []int64
	if len(children) > 4 {
		stepsNode := children[4].gorgoniaNode
		if stepsNode.Dims() > 1 {
			return errors.New("steps node must be a 1-D tensor")
		}
		stepsT, ok := stepsNode.Value().(*tensor.Dense)
		if !ok {
			return errors.New("steps vector must be a dense tensor")
		}
		stepsLen := []int(stepsT.Shape())[0]
		if stepsLen != startsLen {
			return errors.New("all slice tensors must be of the same size")
		}
		steps = make([]int64, stepsLen)
		for i := 0; i < stepsLen; i++ {
			v, err := stepsT.At(0)
			if err != nil {
				return err
			}
			steps[i] = v.(int64)
		}
	} else {
		steps = make([]int64, startsLen)
		for i := 0; i < startsLen; i++ {
			steps[i] = 1
		}
	}
	slices := make([]tensor.Slice, startsLen)
	for i := 0; i < startsLen; i++ {
		slices[i] = rs{
			start: int(starts[i]),
			end:   int(ends[i]),
			step:  int(steps[i]),
		}
	}
	n.gorgoniaNode, err = gorgonia.Slice(children[0].gorgoniaNode, slices...)
	fmt.Println(n.gorgoniaNode.String())

	return err
}

func (s *slice) init(o onnx.Operation) error {
	return nil
}
