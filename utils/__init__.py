import torch


def concat(tensors, retriever=False):
  out = torch.cat(tensors)
  if not retriever:
    return out
  lens = [len(t) for t in tensors]
  def splitter(concated):
    out = []
    for len_ in lens:
      out.append(concated[:len_])
      concated = concated[len_:]
    return out
  return out, splitter
