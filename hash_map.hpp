#pragma once

#include "kmer_t.hpp"
#include <upcxx/upcxx.hpp>

struct HashMap {
    // std::vector<kmer_pair> data;
    // std::vector<int> used;
    std::vector<std::vector<kmer_pair>>* data;
    std::vector<std::vector<int>>* used;

    size_t my_size;
    size_t local_size;

    upcxx::global_ptr<int> used_ptr;
    upcxx::global_ptr<kmer_pair> data_ptr;

    size_t start_index;
    size_t end_index;

    size_t size() const noexcept;

    HashMap(size_t size);

    // Most important functions: insert and retrieve
    // k-mers from the hash table.
    bool insert(const kmer_pair& kmer);
    bool find(const pkmer_t& key_kmer, kmer_pair& val_kmer);

    // Helper functions

    // Write and read to a logical data slot in the table.
    void write_slot(uint64_t slot, const kmer_pair& kmer);
    kmer_pair read_slot(uint64_t slot);

    // Request a slot or check if it's already used.
    bool request_slot(uint64_t slot);
    bool slot_used(uint64_t slot);

    // upcxx::atomic_domain<int> atomics = upcxx::atomic_domain<int>({upcxx::atomic_op::fetch_add,upcxx::atomic_op::load});
    // if using atomic domain destroy? 


};

// added local sizes 
HashMap::HashMap(size_t size) {
    my_size = size;

    local_size = size / upcxx::rank_n();
    start_index = upcxx::rank_me() * local_size;
    end_index = start_index + local_size;
    data = new std::vector<std::vector<kmer_pair>>(local_size);
    used = new std::vector<std::vector<int>>(local_size);
    data_ptr = upcxx::new_array<kmer_pair>(local_size);
    used_ptr = upcxx::new_array<int>(local_size);
    for (size_t i = 0; i < local_size; ++i) {
        data->at(i).resize(1);
        used->at(i).resize(1, 0);
    }
    // data.resize(size);
    // used.resize(size, 0);
}

bool HashMap::insert(const kmer_pair& kmer) {
    uint64_t hash = kmer.hash();
    uint64_t probe = 0;
    bool success = false;
    do {
        uint64_t slot = (hash + probe++) % local_size();
        success = request_slot(slot);
        if (success) {
            write_slot(slot, kmer);
        }
    } while (!success && probe < local_size);
    return success;
}

bool HashMap::find(const pkmer_t& key_kmer, kmer_pair& val_kmer) {
    uint64_t hash = key_kmer.hash();
    uint64_t probe = 0;
    bool success = false;
    // uint64_t current_slot = hash% size();
    do {
        uint64_t slot = (hash + probe++) % local_size();
        if (slot_used(slot)) {
            val_kmer = read_slot(slot);
            if (val_kmer.kmer == key_kmer) {
                success = true;
            }
        }
    } while (!success && probe < local_size);
    return success;
}

bool HashMap::slot_used(uint64_t slot) { upcxx::atomic_rget(used_ptr + slot).wait() != 0; }

void HashMap::write_slot(uint64_t slot, const kmer_pair& kmer) { upcxx::atomic_rput(kmer, data_ptr + slot).wait(); }

kmer_pair HashMap::read_slot(uint64_t slot) { return upcxx::atomic_rget(data_ptr + slot).wait(); }

bool HashMap::request_slot(uint64_t slot) {
    if (upcxx::atomic_rget(used_ptr + slot).wait() != 0) {
        return false;
    } else {
        upcxx::atomic_rput(1, used_ptr + slot).wait();
        return true;
    }
}

size_t HashMap::size() const noexcept { return my_size; }
