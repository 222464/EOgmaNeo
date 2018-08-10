// ----------------------------------------------------------------------------
//  EOgmaNeo
//  Copyright(c) 2017-2018 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of EOgmaNeo is licensed to you under the terms described
//  in the EOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include "ComputeSystem.h"

#include <random>

namespace eogmaneo {
	class EventEncoder;

    /*!
    \brief A single Event.
    */
    struct EventEncoderEvent {
        /*!
        \brief Raveled index of 2D position.
        */
        int _index;

        /*!
        \brief Whether is ON event (true) or OFF event (false).
        */
        bool _polarity;
    };
	
    /*!
    \brief Event encoder work item. Internal use only.
    */
	class EventEncoderActivateWorkItem : public WorkItem {
	public:
		EventEncoder* _pEncoder;

        std::vector<EventEncoderEvent> _events;

		EventEncoderActivateWorkItem()
			: _pEncoder(nullptr)
		{}

		void run(size_t threadIndex) override;
	};
	
    /*!
    \brief Event encoder work item. Internal use only.
    */
	class EventEncoderInhibitWorkItem : public WorkItem {
	public:
		EventEncoder* _pEncoder;

		int _cx, _cy;

		EventEncoderInhibitWorkItem()
			: _pEncoder(nullptr)
		{}

		void run(size_t threadIndex) override;
	};

    /*!
    \brief Event learn work item. Internal use only.
    */
    class EventEncoderLearnWorkItem : public WorkItem {
    public:
        EventEncoder* _pEncoder;

        int _cx, _cy;

        float _alpha;

        EventEncoderLearnWorkItem()
            : _pEncoder(nullptr)
        {}

        void run(size_t threadIndex) override;
    };
	
    /*!
    \brief Encoders values to a columnar SDR through random transformation.
    */
    class EventEncoder {
    private:
        int _inputWidth, _inputHeight;
        int _hiddenWidth, _hiddenHeight;
        int _columnSize;
        int _radius;

        std::vector<int> _hiddenStates;
        std::vector<int> _hiddenActivations;

        std::vector<char> _weights;
        std::vector<float> _biases;

		void activate(const std::vector<EventEncoderEvent> &events);
		void inhibit(int cx, int cy);
        void learn(int cx, int cy, float alpha);
		
    public:
        /*!
        \brief Create the random encoder.
        \param inputWidth input image width.
        \param inputHeight input image height.
        \param hiddenWidth hidden SDR width.
        \param hiddenHeight hidden SDR height.
        \param columnSize column size of hidden SDR.
        \param radius radius onto the input.
        \param seed random number generator seed used when generating this encoder.
        */
        void create(int inputWidth, int inputHeight, int hiddenWidth, int hiddenHeight, int columnSize, int radius,
            unsigned long seed);

        /*!
        \brief Add events. Computes effects of the event passively, only guaranteed when inhibit is called.
        \param cs compute system to be used.
        \param events events to compute.
        */
        void addEvents(ComputeSystem &cs, const std::vector<EventEncoderEvent> &events);

        /*!
        \brief Inhibit the encoder (retrieve SDR formed by events).
        \param cs compute system to be used.
        \return hidden SDR
        */
        const std::vector<int> &inhibit(ComputeSystem &cs);

        /*!
        \brief Learning.
        \param cs compute system to be used.
        \param alpha bias learning rate.
        */
        void learn(ComputeSystem &cs, float alpha);

        //!@{
        /*!
        \brief Get input dimensions.
        */
        int getInputWidth() const {
            return _inputWidth;
        }

        int getInputHeight() const {
            return _inputHeight;
        }
        //!@}

        //!@{
        /*!
        \brief Get hidden dimensions.
        */
        int getHiddenWidth() const {
            return _hiddenWidth;
        }

        int getHiddenHeight() const {
            return _hiddenHeight;
        }
        //!@}

        /*!
        \brief Get (hidden) chunk size.
        */
        int getColumnSize() const {
            return _columnSize;
        }

        /*!
        \brief Get radius of weights onto the input.
        */
        int getRadius() const {
            return _radius;
        }

        /*!
        \brief Get lastly computed hidden states.
        */
        const std::vector<int> &getHiddenStates() const {
            return _hiddenStates;
        }

		friend class EventEncoderActivateWorkItem;
		friend class EventEncoderInhibitWorkItem;
        friend class EventEncoderLearnWorkItem;
    };
}
