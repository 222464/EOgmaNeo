// ----------------------------------------------------------------------------
//  EOgmaNeo
//  Copyright(c) 2017-2018 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of EOgmaNeo is licensed to you under the terms described
//  in the EOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "EventEncoder.h"

#include "Layer.h"

#include <algorithm>
#include <fstream>
#include <assert.h>

using namespace eogmaneo;

void EventEncoderActivateWorkItem::run(size_t threadIndex) {
	_pEncoder->activate(_events);
}

void EventEncoderInhibitWorkItem::run(size_t threadIndex) {
	_pEncoder->inhibit(_cx, _cy);
}

void EventEncoderLearnWorkItem::run(size_t threadIndex) {
    _pEncoder->learn(_cx, _cy, _alpha);
}

void EventEncoder::create(int inputWidth, int inputHeight, int hiddenWidth, int hiddenHeight, int columnSize, int radius,
    unsigned long seed)
{
    std::mt19937 rng;
    rng.seed(seed);

    _inputWidth = inputWidth;
    _inputHeight = inputHeight;
    _hiddenWidth = hiddenWidth;
    _hiddenHeight = hiddenHeight;

    _columnSize = columnSize;

    _radius = radius;

    std::uniform_int_distribution<char> weightDist(-127, 127);

    int diam = _radius * 2 + 1;

    int weightsPerUnit = diam * diam * _columnSize;

	int inUnits = _inputWidth * _inputHeight;

    _weights.resize(inUnits * weightsPerUnit);

    for (int w = 0; w < _weights.size(); w++)
        _weights[w] = weightDist(rng);

    _hiddenStates.resize(_hiddenWidth * _hiddenHeight, 0);
    
    int hUnits = _hiddenStates.size() * _columnSize;

    _biases.resize(hUnits, 0.0f);

    _hiddenActivations.resize(hUnits, 0);
}

void EventEncoder::addEvents(ComputeSystem &cs, const std::vector<EventEncoderEvent> &events) {
    std::shared_ptr<EventEncoderActivateWorkItem> item = std::make_shared<EventEncoderActivateWorkItem>();

    item->_pEncoder = this;
    item->_events = events;

    cs._pool.addItem(item);
}

const std::vector<int> &EventEncoder::inhibit(ComputeSystem &cs) {
    // Wait for spike events
	cs._pool.wait();

    for (int cx = 0; cx < _hiddenWidth; cx++)
        for (int cy = 0; cy < _hiddenHeight; cy++) {
            std::shared_ptr<EventEncoderInhibitWorkItem> item = std::make_shared<EventEncoderInhibitWorkItem>();

            item->_pEncoder = this;
            item->_cx = cx;
            item->_cy = cy;
            
            cs._pool.addItem(item);
        }
        
    cs._pool.wait();

    return _hiddenStates;
}

void EventEncoder::learn(ComputeSystem &cs, float alpha) {
    for (int cx = 0; cx < _hiddenWidth; cx++)
        for (int cy = 0; cy < _hiddenHeight; cy++) {
            std::shared_ptr<EventEncoderLearnWorkItem> item = std::make_shared<EventEncoderLearnWorkItem>();

            item->_pEncoder = this;
            item->_cx = cx;
            item->_cy = cy;
            item->_alpha = alpha;

            cs._pool.addItem(item);
        }

    cs._pool.wait();
}

void EventEncoder::activate(const std::vector<EventEncoderEvent> &events) {
    int diam = _radius * 2 + 1;
    int weightSize = diam * diam;
    int weightsPerUnit = weightSize * _columnSize;

    // Projection
    float toHiddenX = static_cast<float>(_hiddenWidth) / static_cast<float>(_inputWidth);
    float toHiddenY = static_cast<float>(_hiddenHeight) / static_cast<float>(_inputHeight);

    for (int e = 0; e < events.size(); e++) {
        int eIndex = events[e]._index;
        int cx = eIndex % _inputWidth;
        int cy = eIndex / _inputWidth;

        int centerX = cx * toHiddenX + 0.5f;
        int centerY = cy * toHiddenY + 0.5f;

        int lowerX = centerX - _radius;
        int lowerY = centerY - _radius;

        // Compute value
        for (int sx = 0; sx < diam; sx++)
            for (int sy = 0; sy < diam; sy++) {
                int index = sx + sy * diam;

                int hx = lowerX + sx;
                int hy = lowerY + sy;

                if (hx >= 0 && hy >= 0 && hx < _hiddenWidth && hy < _hiddenHeight) {
                    int hi = hx + hy * _hiddenWidth;

                    for (int c = 0; c < _columnSize; c++) {
                        int wi = (index + c * weightSize) + eIndex * weightsPerUnit;
                    
                        _hiddenActivations[hi + c * _hiddenWidth * _hiddenHeight] += events[e]._polarity ? _weights[wi] : -_weights[wi];
                    }
                }
            }
    }
}

void EventEncoder::inhibit(int cx, int cy) {
    int maxCellIndex = 0;
    float maxValue = -99999.0f;

    for (int c = 0; c < _columnSize; c++) {
        int ui = cx + cy * _hiddenWidth + c * _hiddenWidth * _hiddenHeight;

        // Compute value
        float value = _hiddenActivations[ui] + _biases[ui];

        if (value > maxValue) {
            maxValue = value;
            maxCellIndex = c;
        }
    }

	_hiddenStates[cx + cy * _hiddenWidth] = maxCellIndex;
}

void EventEncoder::learn(int cx, int cy, float alpha) {
    for (int c = 0; c < _columnSize; c++) {
        int ui = cx + cy * _hiddenWidth + c * _hiddenWidth * _hiddenHeight;

        _biases[ui] += alpha * (-_hiddenActivations[ui] - _biases[ui]);
    }
}