import math
import torch
from typing import Any, List, Callable, Tuple, Union, NamedTuple
from einops import rearrange


class EasyDict(dict):
    """Convenience class that behaves like a dict but allows access with the attribute syntax."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]

    @staticmethod
    def from_dict(d: dict) -> None:
        """Update the dict with the given dict."""
        obj = EasyDict()
        for k, v in d.items():
            if isinstance(v, dict):
                obj[k] = EasyDict.from_dict(v)
            elif isinstance(v, list) and isinstance(v[0], dict):
                obj[k] = [EasyDict.from_dict(v_) for v_ in v]
            else:
                obj[k] = v
        return obj


class TensorDict(EasyDict):
    """A dict of tensors, potentially with non-tensor values.
    """

    def valid_dic(self) -> dict:
        """Return the valid dict that is not None."""
        return {k: v for k, v in self.items() if v is not None}

    def tensor_keys(self) -> List[str]:
        """Return the tensor keys."""
        return [k for k, v in self.items() if isinstance(v, torch.Tensor)]

    def nontensor_keys(self) -> List[str]:
        """Return the non-tensor keys."""
        return [k for k, v in self.items() if not isinstance(v, torch.Tensor)]

    def tensor_dic(self) -> dict:
        """Return the tensor components in the dict."""
        return {k: v for k, v in self.items() if isinstance(v, torch.Tensor)}

    def nontensor_dic(self) -> dict:
        """Return the non-tensor components in the dict."""
        return {k: v for k, v in self.items() if not isinstance(v, torch.Tensor)}

    def apply(self, func: Callable, *args, **kwargs) -> None:
        """Apply a function to the tensor components."""
        for k, v in self.tensor_dic().items():
            self[k] = func(v, *args, **kwargs)
        return self
    
    def to(self, device):
        """Move all tensors to the device."""
        return self.apply(lambda x: x.to(device))

    def split(self, chunk_size: int = 1, dim: int = 0, squeeze: bool = True) -> List['TensorDict']:
        """Split the tensor components on given dim.
        Args:
            chunk_size: the size of each chunk.
            squeeze: whether to squeeze when chunk_size = 1
        """
        chunks = None
        for k, v in self.tensor_dic().items():
            n_chunk = math.ceil(float(v.shape[0]) / chunk_size)
            if chunks is None:
                chunks = [self.nontensor_dic() for _ in range(n_chunk)]
            for i in range(n_chunk):
                if dim == 0:
                    chunks[i][k] = v[i * chunk_size : (i + 1) * chunk_size]
                elif dim == 1:
                    chunks[i][k] = v[:, i * chunk_size : (i + 1) * chunk_size]
                if squeeze:
                    chunks[i][k] = chunks[i][k].squeeze(dim=dim)
        return [self.__class__(ch) for ch in chunks]

    def rearrange(self, *kargs, **kwargs):
        return self.apply(rearrange, *kargs, **kwargs)

    def detach(self):
        """Detach all tensors."""
        return self.apply(torch.detach)

    def clone(self):
        """Detach all tensors."""
        dic = self.nontensor_dic()
        dic.update({k: v.clone() for k, v in self.tensor_dic().items()})
        return self.__class__(dic)

    def squeeze(self, dim=0):
        """Squeeze all tensors."""
        return self.apply(torch.squeeze, dim=dim)

    def requires_grad_(self, requires_grad=True):
        """Set requires_grad for all tensors."""
        return self.apply(lambda x: x.requires_grad_(requires_grad))

    def slice_between(self, bg, st, dim=0):
        """Slice all tensors between bg and st."""
        dic = self.nontensor_dic()
        dic.update({k: v[bg:st] if dim == 0 else v[:, bg:st]
                    for k, v in self.tensor_dic().items()})
        return self.__class__(dic)

    def slice_indice(self, indice, dim=0):
        """Slice all tensors between bg and st."""
        dic = self.nontensor_dic()
        dic.update({k: v[indice] if dim == 0 else v[:, indice]
                    for k, v in self.tensor_dic().items()})
        return self.__class__(dic)

    @staticmethod
    def list_apply(td_list: List['TensorDict'], func: Callable, *args, **kwargs):
        """Apply a function to the list of tensor dicts."""
        dic = td_list[0].nontensor_dic()
        obj_cls = td_list[0].__class__
        dic.update({k: func([td[k] for td in td_list], *args, **kwargs)
                    for k in td_list[0].tensor_keys()})
        return obj_cls(dic)

    @staticmethod
    def cat(td_list: List['TensorDict'], dim: int = 0):
        """Concatenate the list of tensors."""
        return TensorDict.list_apply(td_list, torch.cat, dim)

    @staticmethod
    def stack(td_list: List['TensorDict'], dim: int = 0):
        """Stack the list of tensors."""
        return TensorDict.list_apply(td_list, torch.stack, dim)