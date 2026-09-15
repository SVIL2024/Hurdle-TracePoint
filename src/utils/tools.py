import torch
import numpy as np

def get_batch_label(texts, prompt_text, label_map: dict):
    label_vectors = torch.zeros(0)
    if len(label_map) != 7:
        if len(label_map) == 2:
            for text in texts:
                label_vector = torch.zeros(2)
                if text == 'Normal':
                    label_vector[0] = 1
                else:
                    label_vector[1] = 1
                label_vector = label_vector.unsqueeze(0)
                label_vectors = torch.cat([label_vectors, label_vector], dim=0)
        else:
            for text in texts:
                label_vector = torch.zeros(len(prompt_text))
                if text in label_map:
                    label_text = label_map[text]
                    label_vector[prompt_text.index(label_text)] = 1

                label_vector = label_vector.unsqueeze(0)
                label_vectors = torch.cat([label_vectors, label_vector], dim=0)
    else:
        for text in texts:
            label_vector = torch.zeros(len(prompt_text))
            labels = text.split('-')
            for label in labels:
                if label in label_map:
                    label_text = label_map[label]
                    label_vector[prompt_text.index(label_text)] = 1

            label_vector = label_vector.unsqueeze(0)
            label_vectors = torch.cat([label_vectors, label_vector], dim=0)

    return label_vectors

def get_prompt_text(label_map: dict):
    prompt_text = []
    for v in label_map.values():
        prompt_text.append(v)

    return prompt_text

def get_batch_mask(lengths, maxlen):
    batch_size = lengths.shape[0]
    mask = torch.empty(batch_size, maxlen)
    mask.fill_(0)
    for i in range(batch_size):
        if lengths[i] < maxlen:
            mask[i, lengths[i]:maxlen] = 1

    return mask.bool()

def random_extract(feat, t_max):
   r = np.random.randint(feat.shape[0] - t_max)
   return feat[r : r+t_max, :]

def uniform_extract(feat, t_max, avg: bool = True):
    new_feat = np.zeros((t_max, feat.shape[1])).astype(np.float32)
    r = np.linspace(0, len(feat), t_max+1, dtype=np.int32)
    if avg == True:
        for i in range(t_max):
            if r[i]!=r[i+1]:
                new_feat[i,:] = np.mean(feat[r[i]:r[i+1],:], 0)
            else:
                new_feat[i,:] = feat[r[i],:]
    else:
        r = np.linspace(0, feat.shape[0]-1, t_max, dtype=np.uint16)
        new_feat = feat[r, :]

    return new_feat

def uniform_extract_with_spans(feat, t_max):
    """Uniformly pool features and retain each pooled token's source span."""
    new_feat = np.zeros((t_max, feat.shape[1]), dtype=np.float32)
    spans = np.zeros(t_max, dtype=np.float32)
    boundaries = np.linspace(0, len(feat), t_max + 1, dtype=np.int32)
    for i in range(t_max):
        start, end = boundaries[i], boundaries[i + 1]
        if start != end:
            new_feat[i, :] = np.mean(feat[start:end, :], axis=0)
            spans[i] = end - start
        else:
            new_feat[i, :] = feat[start, :]
            spans[i] = 1
    return new_feat, spans

def pad(feat, min_len):
    clip_length = feat.shape[0]
    if clip_length <= min_len:
       return np.pad(feat, ((0, min_len - clip_length), (0, 0)), mode='constant', constant_values=0)
    else:
       return feat

def process_feat(feat, length, is_random=False):
    clip_length = feat.shape[0]
    if feat.shape[0] > length:
        if is_random:
            return random_extract(feat, length), length
        else:
            return uniform_extract(feat, length), length
    else:
        return pad(feat, length), clip_length

def process_feat_with_spans(feat, length, is_random=False):
    """Match ``process_feat`` while returning source-span lengths per token."""
    clip_length = feat.shape[0]
    if clip_length > length:
        if is_random:
            start = np.random.randint(clip_length - length)
            sampled = feat[start:start + length, :]
            return sampled, length, np.ones(length, dtype=np.float32)
        sampled, spans = uniform_extract_with_spans(feat, length)
        return sampled, length, spans

    spans = np.zeros(length, dtype=np.float32)
    spans[:clip_length] = 1
    return pad(feat, length), clip_length, spans

def process_split(feat, length):
    clip_length = feat.shape[0]
    if clip_length < length:
        return pad(feat, length), clip_length

    split_num = (clip_length + length - 1) // length
    split_features = []
    for i in range(split_num):
        chunk = feat[i * length:(i + 1) * length, :]
        split_features.append(pad(chunk, length))

    return np.stack(split_features, axis=0), clip_length

def process_split_with_spans(feat, length):
    """Match ``process_split`` while retaining a span vector for each chunk."""
    clip_length = feat.shape[0]
    if clip_length < length:
        spans = np.zeros(length, dtype=np.float32)
        spans[:clip_length] = 1
        return pad(feat, length), clip_length, spans

    split_num = (clip_length + length - 1) // length
    split_features = []
    split_spans = []
    for i in range(split_num):
        chunk = feat[i * length:i * length + length, :]
        valid_length = chunk.shape[0]
        spans = np.zeros(length, dtype=np.float32)
        spans[:valid_length] = 1
        split_features.append(pad(chunk, length))
        split_spans.append(spans)

    return np.stack(split_features, axis=0), clip_length, np.stack(split_spans, axis=0)
