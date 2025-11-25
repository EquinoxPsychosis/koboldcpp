void sample_max_p(llama_token_data_array* cur_p, float p) {
    if (p >= 1.0f) {
        return;
    }

    sample_softmax(cur_p);

    std::vector<llama_token_data> mask;
    std::vector<llama_token_data> uncapped_mask;

    for (size_t i = 0; i < cur_p->size; ++i) {
        if (cur_p->data[i].p > p)
        {
            uncapped_mask.push_back(cur_p->data[i]);
        }
        else
        {
            mask.push_back(cur_p->data[i]);
        }
    }

    if (uncapped_mask.size() == cur_p->size) {
        return;
    }

    float total_excess = 0.0f;
    for (size_t i = 0; i < mask.size(); ++i) {
        float excess = mask[i].p - p;
        total_excess += excess;
    }

    float uncapped_sum = 0.0f;
    for (size_t i = 0; i < uncapped_mask.size(); ++i) {
        uncapped_sum += uncapped_mask[i].p;
    }

    std::vector<llama_token_data> final_probs;

    if (uncapped_sum < 1e-10f) {
        size_t num_tokens = cur_p->size;

        float uniform_prob = 1.0f / static_cast<float>(num_tokens);
        for (size_t i = 0; i < cur_p->size; ++i) {
            cur_p->data[i].logit = std::log(uniform_prob / (1.0f - uniform_prob));
        }

        return;
    }
    else {
        float scale_factor = (uncapped_sum + total_excess) / uncapped_sum;
        for (size_t i = 0; i < mask.size(); ++i) {
            mask[i].p = p;
            mask[i].logit = std::log(mask[i].p / (1.0f - mask[i].p));

            final_probs.push_back(mask[i]);
        }

        for (size_t i = 0; i < uncapped_mask.size(); ++i) {
            uncapped_mask[i].p = uncapped_mask[i].p * scale_factor;
            uncapped_mask[i].logit = std::log(uncapped_mask[i].p / (1.0f - uncapped_mask[i].p));

            final_probs.push_back(uncapped_mask[i]);
        }
    }

    std::copy(final_probs.begin(), final_probs.end(), cur_p->data);

    sample_softmax(cur_p);
}