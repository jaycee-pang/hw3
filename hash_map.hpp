#pragma once

#include "kmer_t.hpp"
#include <upcxx/upcxx.hpp>

struct HashMap {

    // initialize UPC structs for parallelism
    std::vector<upcxx::global_ptr<kmer_pair>> data;     // data array, parallelized with UPC global pointer
    std::vector<upcxx::global_ptr<int>> used;           // used array, parallelized with UPC global pointer
    size_t segment_size;                                // segment size per UPC process
    upcxx::atomic_domain<int32_t>* ad;                  // atomic domain

    size_t my_size;

    size_t size() const noexcept;

    HashMap(size_t size);
    ~HashMap();

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
};

HashMap::HashMap(size_t size) {
    my_size = size;

    // create atomic domain to ensure synchronization when using the HashMap slots
    ad = new upcxx::atomic_domain<int32_t>({upcxx::atomic_op::compare_exchange});

    // calculate segment size for each UPC rank
    int num_procs = upcxx::rank_n();
    int my_rank = upcxx::rank_me();
    segment_size = (my_size + num_procs - 1) / num_procs;
    
    
    // split up data and used arrays across the threads
    data.resize(num_procs);
    used.resize(num_procs);
    size_t segment_start_idx = my_rank * segment_size;
    size_t segment_end_idx = std::min(segment_start_idx + segment_size, my_size);
    data[my_rank] = upcxx::new_array<kmer_pair>(segment_end_idx - segment_start_idx);
    used[my_rank] = upcxx::new_array<int>(segment_end_idx - segment_start_idx);
    
    // broadcast arrays to all the ranks
    for (int i = 0; i < num_procs; ++i) {
        data[i] = upcxx::broadcast(data[i], i).wait();
        used[i] = upcxx::broadcast(used[i], i).wait();
    }

    // create local pointer to used array and fill with all 0s
    int* local_used = used[my_rank].local();
    std::fill_n(local_used, segment_end_idx - segment_start_idx, 0);
}

HashMap::~HashMap() {
    delete ad;
}

bool HashMap::insert(const kmer_pair& kmer) {
    uint64_t hash = kmer.hash();
    uint64_t probe = 0;
    bool success = false;
    do {
        uint64_t slot = (hash + probe++) % size();
        success = request_slot(slot);
        if (success) {
            write_slot(slot, kmer);
        }
    } while (!success && probe < size());
    return success;
}

bool HashMap::find(const pkmer_t& key_kmer, kmer_pair& val_kmer) {
    uint64_t hash = key_kmer.hash();
    uint64_t probe = 0;
    bool success = false;
    do {
        uint64_t slot = (hash + probe++) % size();
        if (slot_used(slot)) {
            val_kmer = read_slot(slot);
            if (val_kmer.kmer == key_kmer) {
                success = true;
            }
        }
    } while (!success && probe < size());
    return success;
}

// In the following functions, now that data and used are parallelized across ranks,
// needed to figure out which rank/index corresponded to which slot numbers

bool HashMap::slot_used(uint64_t slot) {
    int rank = slot / segment_size;
    int idx = slot % segment_size;
    return upcxx::rget(used[rank] + idx).wait() != 0; 
}

void HashMap::write_slot(uint64_t slot, const kmer_pair& kmer) { 
    int rank = slot / segment_size;
    int idx = slot % segment_size;
    return upcxx::rput(kmer, data[rank] + idx).wait();
}

kmer_pair HashMap::read_slot(uint64_t slot) { 
    int rank = slot / segment_size;
    int idx = slot % segment_size;
    return upcxx::rget(data[rank] + idx).wait(); 
}

bool HashMap::request_slot(uint64_t slot) {
    int rank = slot / segment_size;
    int idx = slot % segment_size;

    // use compare_exchange to determine if a slot is being used or not
    // and then take the slot if it's not being used
    int expected = 0;
    int desired = 1;
    int found = ad->compare_exchange(used[rank] + idx, expected, desired, std::memory_order_relaxed).wait();
    
    if (found != 0) {
        return false;
    } else {
        return true;
    }
}

size_t HashMap::size() const noexcept { return my_size; }