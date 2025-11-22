void sample_top_h(llama_token_data_array * cur_p, float coef, size_t top_n) {
    if ((coef < 0.0f || coef >= 1.0f) || cur_p->size < 2) {
        return;
    }

    top_n = std::min(top_n, cur_p->size);

    // Sort and normalize logits
    sample_softmax(cur_p);

    float alpha = 0.0f;
    // Calculate sum of probabilities of the top 100 most probable tokens
    for (size_t i = 0; i < top_n; ++i) {
        alpha += cur_p->data[i].p;
    }

    float entropy = 0.0f;
    // Calculate entropy
    for (size_t i = 0; i < top_n; ++i) {
        float entropy_a = cur_p->data[i].p / alpha;
        entropy -= entropy_a * log2(entropy_a);
    }

    // Caclulate tau
    float tau = ((entropy - log2(alpha)) * alpha) * coef;

    size_t keep_tokens = 0;
    float  sigma       = cur_p->data[0].p;
    float  H           = -cur_p->data[0].p * log2(cur_p->data[0].p);
    // Find which tokens to keep from entropy thresholding
    for (size_t i = 0; i < top_n; ++i) {
        keep_tokens++;

        float next_p = cur_p->data[i + 1].p;
        H -= next_p * log2(next_p);
        sigma += next_p;

        float entropy_diff = ((H / sigma) + log2(sigma));
        float threshold    = (tau / sigma + log2(sigma));

        if (entropy_diff > threshold) {
            break;
        }
    }

    // Trim down to keep_tokens mask
    cur_p->size = keep_tokens;
}