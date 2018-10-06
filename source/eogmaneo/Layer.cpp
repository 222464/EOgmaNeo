// ----------------------------------------------------------------------------
//  EOgmaNeo
//  Copyright(c) 2017-2018 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of EOgmaNeo is licensed to you under the terms described
//  in the EOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "Layer.h"

#include <algorithm>
#include <thread>
#include <future>
#include <iostream>

#include <assert.h>

using namespace eogmaneo;

float eogmaneo::sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

void LayerForwardWorkItem::run() {
	_pLayer->columnForward(_ci);
}

void LayerBackwardWorkItem::run() {
	_pLayer->columnBackward(_ci, _v, _rng);
}

void Layer::columnForward(int ci) {
    int hiddenColumnX = ci % _hiddenWidth;
    int hiddenColumnY = ci / _hiddenWidth;

    int hiddenStatePrev = _hiddenStatesPrev[ci];

    int hiddenCellIndexPrev = ci + hiddenStatePrev * _hiddenWidth * _hiddenHeight;

    std::vector<float> columnActivations(_columnSize, 0.0f);

    // Activate feed forward
    for (int v = 0; v < _visibleLayerDescs.size(); v++) {
        float toInputX = static_cast<float>(_visibleLayerDescs[v]._width) / static_cast<float>(_hiddenWidth);
        float toInputY = static_cast<float>(_visibleLayerDescs[v]._height) / static_cast<float>(_hiddenHeight);

        int visibleCenterX = hiddenColumnX * toInputX + 0.5f;
        int visibleCenterY = hiddenColumnY * toInputY + 0.5f;

        int forwardRadius = _visibleLayerDescs[v]._forwardRadius;

        int forwardDiam = forwardRadius * 2 + 1;

        int forwardSize = forwardDiam * forwardDiam;

        int lowerVisibleX = visibleCenterX - forwardRadius;
        int lowerVisibleY = visibleCenterY - forwardRadius;

        for (int dcx = -forwardRadius; dcx <= forwardRadius; dcx++)
            for (int dcy = -forwardRadius; dcy <= forwardRadius; dcy++) {
                int cx = visibleCenterX + dcx;
                int cy = visibleCenterY + dcy;

                if (cx >= 0 && cx < _visibleLayerDescs[v]._width && cy >= 0 && cy < _visibleLayerDescs[v]._height) {
                    int visibleColumnIndex = cx + cy * _visibleLayerDescs[v]._width;

                    int inputIndex = _inputs[v][visibleColumnIndex];
                    int inputIndexPrev = _inputsPrev[v][visibleColumnIndex];

                    if (_codeIter == 0 && _learn && !_reconsActLearn.empty()) {
                        // Input cells
                        for (int c = 0; c < _visibleLayerDescs[v]._columnSize; c++) {
                            int wi = (cx - lowerVisibleX) + (cy - lowerVisibleY) * forwardDiam + c * forwardSize;

                            int visibleCellIndex = visibleColumnIndex + c * _visibleLayerDescs[v]._width * _visibleLayerDescs[v]._height;

                            float recon = _reconsActLearn[v][visibleCellIndex] / std::max(1.0f, _reconCountsActLearn[v][visibleCellIndex]);

                            float target = (c == inputIndexPrev ? 1.0f : 0.0f);
                            
                            _feedForwardWeights[v][hiddenCellIndexPrev][wi] = std::max(0.0f, _feedForwardWeights[v][hiddenCellIndexPrev][wi] + _alpha * (target - recon));
                        }
                    }

                    // Output cells
                    if (_codeIter == 0) {
                        int wi = (cx - lowerVisibleX) + (cy - lowerVisibleY) * forwardDiam + inputIndex * forwardSize;

                        for (int c = 0; c < _columnSize; c++) {
                            int hiddenCellIndex = ci + c * _hiddenWidth * _hiddenHeight;
                            
                            columnActivations[c] += _feedForwardWeights[v][hiddenCellIndex][wi];
                        }
                    }
                    else {
                        int wi = (cx - lowerVisibleX) + (cy - lowerVisibleY) * forwardDiam + inputIndex * forwardSize;

                        int visibleCellIndex = visibleColumnIndex + inputIndex * _visibleLayerDescs[v]._width * _visibleLayerDescs[v]._height;

                        float recon = _reconsActLearn[v][visibleCellIndex] / std::max(1.0f, _reconCountsActLearn[v][visibleCellIndex]);

                        for (int c = 0; c < _columnSize; c++) {
                            int hiddenCellIndex = ci + c * _hiddenWidth * _hiddenHeight;
                            
                            columnActivations[c] += std::max(0.0f, _feedForwardWeights[v][hiddenCellIndex][wi] - recon);
                        }
                    }
                }
            }
    }

	// Find max element
	int maxCellIndex = 0;
    float maxValue = -99999.0f;

	for (int c = 0; c < _columnSize; c++) {
        int hiddenCellIndex = ci + c * _hiddenWidth * _hiddenHeight;

        if (_codeIter == 0)
            _hiddenActivations[hiddenCellIndex] = columnActivations[c];
        else
            _hiddenActivations[hiddenCellIndex] += columnActivations[c];

		if (_hiddenActivations[hiddenCellIndex] > maxValue) {
            maxValue = _hiddenActivations[hiddenCellIndex];
			maxCellIndex = c;
        }
	}

    _hiddenStates[ci] = maxCellIndex;

    int hiddenCellIndex = ci + maxCellIndex * _hiddenWidth * _hiddenHeight;

    // Reconstruct
    for (int v = 0; v < _visibleLayerDescs.size(); v++) {
        float toInputX = static_cast<float>(_visibleLayerDescs[v]._width) / static_cast<float>(_hiddenWidth);
        float toInputY = static_cast<float>(_visibleLayerDescs[v]._height) / static_cast<float>(_hiddenHeight);

        int visibleCenterX = hiddenColumnX * toInputX + 0.5f;
        int visibleCenterY = hiddenColumnY * toInputY + 0.5f;

        int forwardRadius = _visibleLayerDescs[v]._forwardRadius;

        int forwardDiam = forwardRadius * 2 + 1;

        int forwardSize = forwardDiam * forwardDiam;

        int lowerVisibleX = visibleCenterX - forwardRadius;
        int lowerVisibleY = visibleCenterY - forwardRadius;

        for (int dcx = -forwardRadius; dcx <= forwardRadius; dcx++)
            for (int dcy = -forwardRadius; dcy <= forwardRadius; dcy++) {
                int cx = visibleCenterX + dcx;
                int cy = visibleCenterY + dcy;

                if (cx >= 0 && cx < _visibleLayerDescs[v]._width && cy >= 0 && cy < _visibleLayerDescs[v]._height) {
                    int visibleColumnIndex = cx + cy * _visibleLayerDescs[v]._width;

                    // Input cells
                    for (int c = 0; c < _visibleLayerDescs[v]._columnSize; c++) {
                        int wi = (cx - lowerVisibleX) + (cy - lowerVisibleY) * forwardDiam + c * forwardSize;

                        int visibleCellIndex = visibleColumnIndex + c * _visibleLayerDescs[v]._width * _visibleLayerDescs[v]._height;

                        _recons[v][visibleCellIndex] += _feedForwardWeights[v][hiddenCellIndex][wi];
                        _reconCounts[v][visibleCellIndex] += 1.0f;
                    }
                }
            }
    }
}

void Layer::columnBackward(int ci, int v, std::mt19937 &rng) {
    int visibleWidth = _visibleLayerDescs[v]._width;
    int visibleHeight = _visibleLayerDescs[v]._height;

    int visibleColumnX = ci % visibleWidth;
    int visibleColumnY = ci / visibleWidth;

    int visibleColumnSize = _visibleLayerDescs[v]._columnSize;
    
    int backwardRadius = _visibleLayerDescs[v]._backwardRadius;

    int backwardDiam = backwardRadius * 2 + 1;
    int backwardSize = backwardDiam * backwardDiam;
    int backwardVecSize = backwardSize * _columnSize;

    float toInputX = static_cast<float>(_hiddenWidth) / static_cast<float>(visibleWidth);
    float toInputY = static_cast<float>(_hiddenHeight) / static_cast<float>(visibleHeight);

    int hiddenCenterX = visibleColumnX * toInputX + 0.5f;
    int hiddenCenterY = visibleColumnY * toInputY + 0.5f;

    int lowerHiddenX = hiddenCenterX - backwardRadius;
    int lowerHiddenY = hiddenCenterY - backwardRadius;

    std::vector<float> columnActivations(visibleColumnSize, 0.0f);
    std::vector<float> columnActivationsPrev(visibleColumnSize, 0.0f);
    float count = 0.0f;

    int updateIndex = _predictions[v][ci];

    int visibleCellIndexUpdate = ci + updateIndex * visibleWidth * visibleHeight;

    for (int dcx = -backwardRadius; dcx <= backwardRadius; dcx++)
        for (int dcy = -backwardRadius; dcy <= backwardRadius; dcy++) {
            int cx = hiddenCenterX + dcx;
            int cy = hiddenCenterY + dcy;

            if (cx >= 0 && cx < _hiddenWidth && cy >= 0 && cy < _hiddenHeight) {
                int hiddenColumnIndex = cx + cy * _hiddenWidth;

                if (!_feedBack.empty() && !_feedBackPrev.empty()) {
                    int feedBackIndex = _feedBack[hiddenColumnIndex];
                    int feedBackIndexPrev = _feedBackPrev[hiddenColumnIndex];

                    // Output cells
                    int wiCur = (cx - lowerHiddenX) + (cy - lowerHiddenY) * backwardDiam + feedBackIndex * backwardSize;
                    int wiPrev = (cx - lowerHiddenX) + (cy - lowerHiddenY) * backwardDiam + feedBackIndexPrev * backwardSize;

                    for (int c = 0; c < visibleColumnSize; c++) {
                        int visibleCellIndex = ci + c * visibleWidth * visibleHeight;
                            
                        columnActivations[c] += _feedBackWeights[v][visibleCellIndex][wiCur];
                        columnActivationsPrev[c] += _feedBackWeights[v][visibleCellIndex][wiPrev];
                    }

                    count += 1.0f;
                }

                int hiddenIndex = _hiddenStates[hiddenColumnIndex];
                int hiddenIndexPrev = _hiddenStatesPrev[hiddenColumnIndex];

                int wiCur = (cx - lowerHiddenX) + (cy - lowerHiddenY) * backwardDiam + hiddenIndex * backwardSize + backwardVecSize;
                int wiPrev = (cx - lowerHiddenX) + (cy - lowerHiddenY) * backwardDiam + hiddenIndexPrev * backwardSize + backwardVecSize;
                
                // Output cells
                for (int c = 0; c < visibleColumnSize; c++) {
                    int visibleCellIndex = ci + c * visibleWidth * visibleHeight;
        
                    columnActivations[c] += _feedBackWeights[v][visibleCellIndex][wiCur];
                    columnActivationsPrev[c] += _feedBackWeights[v][visibleCellIndex][wiPrev];
                }

                count += 1.0f;
            }
        }

    float rescale = 1.0f / std::max(1.0f, count);

    int maxIndex = 0;
    int maxIndexPrev = 0;

    for (int c = 0; c < visibleColumnSize; c++) {
        columnActivations[c] *= rescale;
        columnActivationsPrev[c] *= rescale;

        if (columnActivations[c] > columnActivations[maxIndex])
            maxIndex = c;

        if (columnActivationsPrev[c] > columnActivationsPrev[maxIndexPrev])
            maxIndexPrev = c;
    }

    std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

    if (dist01(rng) < _epsilon) {
        std::uniform_int_distribution<int> columnDist(0, visibleColumnSize - 1);

        _predictions[v][ci] = columnDist(rng);
    }
    else
        _predictions[v][ci] = maxIndex;

    float dQ = _reward + _gamma * columnActivations[maxIndex] - columnActivationsPrev[updateIndex];

    float dAdv = dQ - _k * (columnActivationsPrev[maxIndexPrev] - columnActivationsPrev[updateIndex]);

    float dPAL = std::max(dAdv, dQ - _k * (columnActivations[maxIndex] - columnActivations[updateIndex]));
    
    float update = _beta * dPAL;
        
    for (int dcx = -backwardRadius; dcx <= backwardRadius; dcx++)
        for (int dcy = -backwardRadius; dcy <= backwardRadius; dcy++) {
            int cx = hiddenCenterX + dcx;
            int cy = hiddenCenterY + dcy;

            if (cx >= 0 && cx < _hiddenWidth && cy >= 0 && cy < _hiddenHeight) {
                int hiddenColumnIndex = cx + cy * _hiddenWidth;

                if (!_feedBack.empty()) {
                    int feedBackIndexPrev = _feedBackPrev[hiddenColumnIndex];

                    // Output cells
                    int wiPrev = (cx - lowerHiddenX) + (cy - lowerHiddenY) * backwardDiam + feedBackIndexPrev * backwardSize;

                    _feedBackWeights[v][visibleCellIndexUpdate][wiPrev] += update;
                }

                int hiddenIndexPrev = _hiddenStatesPrev[hiddenColumnIndex];

                int wiPrev = (cx - lowerHiddenX) + (cy - lowerHiddenY) * backwardDiam + hiddenIndexPrev * backwardSize + backwardVecSize;

                _feedBackWeights[v][visibleCellIndexUpdate][wiPrev] += update;
            }
        }
}

void Layer::create(int hiddenWidth, int hiddenHeight, int columnSize, const std::vector<VisibleLayerDesc> &visibleLayerDescs, unsigned long seed) {
    std::mt19937 rng(seed);

    _hiddenWidth = hiddenWidth;
    _hiddenHeight = hiddenHeight;
    _columnSize = columnSize;

    _visibleLayerDescs = visibleLayerDescs;

    _feedForwardWeights.resize(_visibleLayerDescs.size());
    _feedBackWeights.resize(_visibleLayerDescs.size());

    _inputs.resize(_visibleLayerDescs.size());

    _hiddenStates.resize(_hiddenWidth * _hiddenHeight, 0);

    _hiddenActivations.resize(_hiddenStates.size() * _columnSize, 0.0f);
    
    std::uniform_real_distribution<float> initWeightDistHigh(0.99f, 1.0f);
    std::uniform_real_distribution<float> initWeightDistLow(-0.0001f, 0.0001f);

    for (int v = 0; v < _visibleLayerDescs.size(); v++) {
        _inputs[v].resize(_visibleLayerDescs[v]._width * _visibleLayerDescs[v]._height, 0);

        int forwardVecSize = _visibleLayerDescs[v]._forwardRadius * 2 + 1;

        forwardVecSize *= forwardVecSize * _visibleLayerDescs[v]._columnSize;

        _feedForwardWeights[v].resize(_hiddenWidth * _hiddenHeight * _columnSize);

        for (int x = 0; x < _hiddenWidth; x++)
            for (int y = 0; y < _hiddenHeight; y++)
                for (int c = 0; c < _columnSize; c++) {
                    int hiddenCellIndex = x + y * _hiddenWidth + c * _hiddenWidth * _hiddenHeight;

                    _feedForwardWeights[v][hiddenCellIndex].resize(forwardVecSize);
                    
                    for (int j = 0; j < forwardVecSize; j++)
                        _feedForwardWeights[v][hiddenCellIndex][j] = initWeightDistHigh(rng);
                }

        if (_visibleLayerDescs[v]._predict) {
            _feedBackWeights[v].resize(_visibleLayerDescs[v]._width * _visibleLayerDescs[v]._height * _visibleLayerDescs[v]._columnSize);

            int backwardVecSize = _visibleLayerDescs[v]._backwardRadius * 2 + 1;

            backwardVecSize *= backwardVecSize * _columnSize * 2;

            for (int x = 0; x < _visibleLayerDescs[v]._width; x++)
                for (int y = 0; y < _visibleLayerDescs[v]._height; y++)         
                    for (int c = 0; c < _visibleLayerDescs[v]._columnSize; c++) {
                        int visibleCellIndex = x + y * _visibleLayerDescs[v]._width + c * _visibleLayerDescs[v]._width * _visibleLayerDescs[v]._height;
                        
                        _feedBackWeights[v][visibleCellIndex].resize(backwardVecSize);

                        for (int j = 0; j < backwardVecSize; j++)
                            _feedBackWeights[v][visibleCellIndex][j] = initWeightDistLow(rng);
                    }
        }
    }

    _feedBackPrev = _feedBack = _hiddenStatesPrev = _hiddenStates;

    _predictions = _inputsPrev = _inputs;
}

void Layer::forward(ComputeSystem &cs, const std::vector<std::vector<int>> &inputs, bool learn) {
    _inputsPrev = _inputs;
    _inputs = inputs;

    _learn = learn;

    _hiddenStatesPrev = _hiddenStates;

    // Several inhibition iterations
    for (int it = 0; it < _codeIters; it++) {
        _codeIter = it;

        // Clear recons
        _recons.clear();
        _recons.resize(_visibleLayerDescs.size());

        for (int v = 0; v < _visibleLayerDescs.size(); v++)
            _recons[v].resize(_visibleLayerDescs[v]._width * _visibleLayerDescs[v]._height * _visibleLayerDescs[v]._columnSize, 0.0f);
        
        _reconCounts = _recons;

        for (int ci = 0; ci < _hiddenStates.size(); ci++) {
            std::shared_ptr<LayerForwardWorkItem> item = std::make_shared<LayerForwardWorkItem>();

            item->_pLayer = this;
            item->_ci = ci;

            cs._pool.addItem(item);
        }
        
        cs._pool.wait();

        _reconsActLearn = _recons;
        _reconCountsActLearn = _reconCounts;
    }
}

void Layer::backward(ComputeSystem &cs, const std::vector<int> &feedBack, float reward, bool learn) {
    _feedBackPrev = _feedBack;
	_feedBack = feedBack;

    _learn = learn;

    _reward = reward;

    std::uniform_int_distribution<int> seedDist(0, 99999);

    for (int v = 0; v < _visibleLayerDescs.size(); v++) {
        if (!_visibleLayerDescs[v]._predict)
            continue;

        for (int ci = 0; ci < _predictions[v].size(); ci++) {
            std::shared_ptr<LayerBackwardWorkItem> item = std::make_shared<LayerBackwardWorkItem>();

            item->_pLayer = this;
            item->_ci = ci;
            item->_v = v;
            item->_rng.seed(seedDist(cs._rng));

            cs._pool.addItem(item);
        }
    }

    cs._pool.wait();
}

void Layer::readFromStream(std::istream &is) {
    // Read header
    is.read(reinterpret_cast<char*>(&_hiddenWidth), sizeof(int));
    is.read(reinterpret_cast<char*>(&_hiddenHeight), sizeof(int));
    is.read(reinterpret_cast<char*>(&_columnSize), sizeof(int));

    // Read hyperparameters
    is.read(reinterpret_cast<char*>(&_alpha), sizeof(float));
    is.read(reinterpret_cast<char*>(&_beta), sizeof(float));
    is.read(reinterpret_cast<char*>(&_gamma), sizeof(float));
    is.read(reinterpret_cast<char*>(&_k), sizeof(float));
    is.read(reinterpret_cast<char*>(&_epsilon), sizeof(float));
    is.read(reinterpret_cast<char*>(&_codeIters), sizeof(int));

    int numVisibleLayerDescs;

    is.read(reinterpret_cast<char*>(&numVisibleLayerDescs), sizeof(int));

    _visibleLayerDescs.resize(numVisibleLayerDescs);

    is.read(reinterpret_cast<char*>(_visibleLayerDescs.data()), _visibleLayerDescs.size() * sizeof(VisibleLayerDesc));

    _inputs.resize(_visibleLayerDescs.size());
    _inputsPrev.resize(_visibleLayerDescs.size());
    _predictions.resize(_visibleLayerDescs.size());

    _feedForwardWeights.resize(_visibleLayerDescs.size());
    _feedBackWeights.resize(_visibleLayerDescs.size());
   
    // Hidden data
    _hiddenStates.resize(_hiddenWidth * _hiddenHeight);
    _hiddenStatesPrev.resize(_hiddenStates.size());
    _feedBack.resize(_hiddenStates.size());
    _feedBackPrev.resize(_hiddenStates.size());
    _hiddenActivations.resize(_hiddenStates.size() * _columnSize, 0.0f);

    is.read(reinterpret_cast<char*>(_hiddenStates.data()), _hiddenStates.size() * sizeof(int));
    is.read(reinterpret_cast<char*>(_hiddenStatesPrev.data()), _hiddenStatesPrev.size() * sizeof(int));
    is.read(reinterpret_cast<char*>(_feedBack.data()), _feedBack.size() * sizeof(int));
    is.read(reinterpret_cast<char*>(_feedBackPrev.data()), _feedBackPrev.size() * sizeof(int));

    // If feedback is -1, clear to empty
    if (_feedBack.front() == -1)
        _feedBack.clear();

    if (_feedBackPrev.front() == -1)
        _feedBackPrev.clear();

    for (int v = 0; v < _visibleLayerDescs.size(); v++) {
        // Visible layer data
        _inputs[v].resize(_visibleLayerDescs[v]._width * _visibleLayerDescs[v]._height);
        _inputsPrev[v].resize(_inputs[v].size());
        _predictions[v].resize(_inputs[v].size());
        
        is.read(reinterpret_cast<char*>(_inputs[v].data()), _inputs[v].size() * sizeof(int));
        is.read(reinterpret_cast<char*>(_inputsPrev[v].data()), _inputsPrev[v].size() * sizeof(int));
        is.read(reinterpret_cast<char*>(_predictions[v].data()), _predictions[v].size() * sizeof(int));

        // Forward weights
        int forwardVecSize = _visibleLayerDescs[v]._forwardRadius * 2 + 1;

        forwardVecSize *= forwardVecSize * _visibleLayerDescs[v]._columnSize;

        _feedForwardWeights[v].resize(_hiddenWidth * _hiddenHeight * _columnSize);

        for (int x = 0; x < _hiddenWidth; x++)
            for (int y = 0; y < _hiddenHeight; y++)
                for (int c = 0; c < _columnSize; c++) {
                    int hiddenCellIndex = x + y * _hiddenWidth + c * _hiddenWidth * _hiddenHeight;

                    _feedForwardWeights[v][hiddenCellIndex].resize(forwardVecSize);

                    is.read(reinterpret_cast<char*>(_feedForwardWeights[v][hiddenCellIndex].data()), _feedForwardWeights[v][hiddenCellIndex].size() * sizeof(float));
                }

        // Backward weights
        if (_visibleLayerDescs[v]._predict) {
            _feedBackWeights[v].resize(_visibleLayerDescs[v]._width * _visibleLayerDescs[v]._height * _visibleLayerDescs[v]._columnSize);

            int backwardVecSize = _visibleLayerDescs[v]._backwardRadius * 2 + 1;

            backwardVecSize *= backwardVecSize * _columnSize;

            for (int x = 0; x < _visibleLayerDescs[v]._width; x++)
                for (int y = 0; y < _visibleLayerDescs[v]._height; y++)         
                    for (int c = 0; c < _visibleLayerDescs[v]._columnSize; c++) {
                        int visibleCellIndex = x + y * _visibleLayerDescs[v]._width + c * _visibleLayerDescs[v]._width * _visibleLayerDescs[v]._height;

                        _feedBackWeights[v][visibleCellIndex].resize(backwardVecSize);
                            
                        is.read(reinterpret_cast<char*>(_feedBackWeights[v][visibleCellIndex].data()), _feedBackWeights[v][visibleCellIndex].size() * sizeof(float));
                    }
        }
    }
}

void Layer::writeToStream(std::ostream &os) {
    // Write header
    os.write(reinterpret_cast<char*>(&_hiddenWidth), sizeof(int));
    os.write(reinterpret_cast<char*>(&_hiddenHeight), sizeof(int));
    os.write(reinterpret_cast<char*>(&_columnSize), sizeof(int));

    // Write hyperparameters
    os.write(reinterpret_cast<char*>(&_alpha), sizeof(float));
    os.write(reinterpret_cast<char*>(&_beta), sizeof(float));
    os.write(reinterpret_cast<char*>(&_gamma), sizeof(float));
    os.write(reinterpret_cast<char*>(&_k), sizeof(float));
    os.write(reinterpret_cast<char*>(&_epsilon), sizeof(float));
    os.write(reinterpret_cast<char*>(&_codeIters), sizeof(int));

    int numVisibleLayerDescs = _visibleLayerDescs.size();

    os.write(reinterpret_cast<char*>(&numVisibleLayerDescs), sizeof(int));

    os.write(reinterpret_cast<char*>(_visibleLayerDescs.data()), _visibleLayerDescs.size() * sizeof(VisibleLayerDesc));

    // Hidden data
    os.write(reinterpret_cast<char*>(_hiddenStates.data()), _hiddenStates.size() * sizeof(int));
    os.write(reinterpret_cast<char*>(_hiddenStatesPrev.data()), _hiddenStatesPrev.size() * sizeof(int));

    std::vector<int> writeFeedBack = _feedBack;
    std::vector<int> writeFeedBackPrev = _feedBackPrev;

    if (writeFeedBack.empty())
        writeFeedBack.resize(_hiddenStates.size(), -1);

    if (writeFeedBackPrev.empty())
        writeFeedBackPrev.resize(_hiddenStates.size(), -1);

    os.write(reinterpret_cast<char*>(writeFeedBack.data()), writeFeedBack.size() * sizeof(int));
    os.write(reinterpret_cast<char*>(writeFeedBackPrev.data()), writeFeedBackPrev.size() * sizeof(int));

    for (int v = 0; v < _visibleLayerDescs.size(); v++) {
        // Visible layer data
        os.write(reinterpret_cast<char*>(_inputs[v].data()), _inputs[v].size() * sizeof(int));
        os.write(reinterpret_cast<char*>(_inputsPrev[v].data()), _inputsPrev[v].size() * sizeof(int));
        os.write(reinterpret_cast<char*>(_predictions[v].data()), _predictions[v].size() * sizeof(int));

        // Forward weights
        for (int x = 0; x < _hiddenWidth; x++)
            for (int y = 0; y < _hiddenHeight; y++)
                for (int c = 0; c < _columnSize; c++) {
                    int hiddenCellIndex = x + y * _hiddenWidth + c * _hiddenWidth * _hiddenHeight;

                    os.write(reinterpret_cast<char*>(_feedForwardWeights[v][hiddenCellIndex].data()), _feedForwardWeights[v][hiddenCellIndex].size() * sizeof(float));
                }

        // Backward weights
        if (_visibleLayerDescs[v]._predict) {
            for (int x = 0; x < _visibleLayerDescs[v]._width; x++)
                for (int y = 0; y < _visibleLayerDescs[v]._height; y++)         
                    for (int c = 0; c < _visibleLayerDescs[v]._columnSize; c++) {
                        int visibleCellIndex = x + y * _visibleLayerDescs[v]._width + c * _visibleLayerDescs[v]._width * _visibleLayerDescs[v]._height;
                            
                        os.write(reinterpret_cast<char*>(_feedBackWeights[v][visibleCellIndex].data()), _feedBackWeights[v][visibleCellIndex].size() * sizeof(float));
                    }
        }
    }
}

void Layer::zeroContext() {
    for (int v = 0; v < _visibleLayerDescs.size(); v++) {
        for (int i = 0; i < _inputs[v].size(); i++) {
            _inputs[v][i] = 0;
            _inputsPrev[v][i] = 0;
            _predictions[v][i] = 0;
        }
    }

    for (int i = 0; i < _hiddenStates.size(); i++) {
        _hiddenStates[i] = 0;
        _hiddenStatesPrev[i] = 0;
    }
}