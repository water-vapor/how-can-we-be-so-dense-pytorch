import torch


def random_binary_mask(tensor_shape, false_percentage, strict=True):
    if not strict:
        binary_mask = torch.rand(tensor_shape) > false_percentage
    else:
        num_elements = torch.prod(torch.tensor(tensor_shape))
        binary_mask = torch.reshape(
            torch.linspace(0.0, 1.0, num_elements)[torch.randperm(num_elements)]
            > false_percentage,
            tensor_shape,
        )
    return binary_mask


def kwinner_mask(inputs, k):
    vals, _ = torch.topk(inputs, k=k)
    kth_elems, _ = torch.min(vals, dim=-1, keepdim=True)
    boolean_mask = inputs >= kth_elems
    return boolean_mask


def flat_kwinner_mask(inputs, k):
    if len(inputs.shape) > 2:
        input_shape_batchless = inputs.shape[1:]
        flat_input = torch.reshape(inputs, (-1, torch.prod(torch.tensor(input_shape_batchless))))
        boolean_mask = kwinner_mask(flat_input, k)
        return torch.reshape(boolean_mask, (-1, *input_shape_batchless))
    else:
        return kwinner_mask(inputs, k)


def add_noise(img, eta, strict=True):
    noise_val = torch.mean(img) + 2 * torch.std(img)
    binary_mask = random_binary_mask(img.shape, eta, strict)
    return torch.where(binary_mask, img, noise_val)
