#pragma once

#include "tree/IPruner.hpp"

class MinGainPrePruner : public IPruner {
public:
    explicit MinGainPrePruner(double minGain) : minGain_(minGain) {}
    
    void prune(std::unique_ptr<Node>&) const override {}   // Pre-pruning happens during training
    double minGain() const { return minGain_; }
private:
    double minGain_;
};