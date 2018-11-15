import torch.nn as nn
import torch


class ZoneoutLSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, zoneout_factor_cell=0.1, zoneout_factor_hidden=0.1, bidirectional=True):
        super(ZoneoutLSTMEncoder, self).__init__()
        if zoneout_factor_cell < 0. or zoneout_factor_cell > 1. \
                or zoneout_factor_hidden < 0. or zoneout_factor_hidden > 1.:
            raise ValueError("'One/both provided Zoneout factors are not in [0, 1]")

        self.lstm_cell = nn.LSTMCell(input_size=input_size, hidden_size=hidden_size)
        self.zoneout_cell = zoneout_factor_cell
        self.zoneout_hidden = zoneout_factor_hidden
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional

    def forward(self, inputs, hidden_state=None, cell_state=None, is_training=True):
        """
        :param inputs: T x B x C
        :param state:
        :param is_training:
        :return:
        """
        assert self.bidirectional
        seq, batch_size, inputs_size = inputs.size()
        hidden_state_0 = torch.zeros((batch_size, self.hidden_size)).cuda(0) if hidden_state is None else hidden_state
        cell_state_0 = torch.zeros((batch_size, self.hidden_size)).cuda(0) if cell_state is None else cell_state
        hidden_state_1 = torch.zeros((batch_size, self.hidden_size)).cuda(0) if hidden_state is None else hidden_state
        cell_state_1 = torch.zeros((batch_size, self.hidden_size)).cuda(0) if cell_state is None else cell_state

        outputs_0_list = []
        outputs_1_list = []

        for i in range(seq):
            new_hidden_state_0, new_cell_state_0 = self.lstm_cell(inputs[i], (hidden_state_0, cell_state_0))
            if is_training:
                cell_state_0 = (1 - self.zoneout_cell) * \
                    nn.functional.dropout(new_cell_state_0 - cell_state_0, (1 - self.zoneout_cell)) + cell_state_0
                hidden_state_0 = (1 - self.zoneout_hidden) * \
                    nn.functional.dropout(new_hidden_state_0 - hidden_state_0, (1 - self.zoneout_hidden)) + \
                    hidden_state_0
            else:
                cell_state_0 = (1 - self.zoneout_cell) * new_cell_state_0 + self.zoneout_cell * cell_state_0
                hidden_state_0 = (1 - self.zoneout_hidden) * new_hidden_state_0 + self.zoneout_hidden * hidden_state_0
            outputs_0_list.append(hidden_state_0)

            new_hidden_state_1, new_cell_state_1 = self.lstm_cell(inputs[seq-1-i], (hidden_state_1, cell_state_1))
            if is_training:
                cell_state_1 = (1 - self.zoneout_cell) * \
                    nn.functional.dropout(new_cell_state_1 - cell_state_1, (1 - self.zoneout_cell)) + cell_state_1
                hidden_state_1 = (1 - self.zoneout_hidden) * \
                    nn.functional.dropout(new_hidden_state_1 - hidden_state_1, (1 - self.zoneout_hidden)) + \
                    hidden_state_1
            else:
                cell_state_1 = (1 - self.zoneout_cell) * new_cell_state_1 + self.zoneout_cell * cell_state_1
                hidden_state_1 = (1 - self.zoneout_hidden) * new_hidden_state_1 + self.zoneout_hidden * hidden_state_1

            outputs_1_list.insert(0, hidden_state_1)
        outputs_0 = torch.stack(outputs_0_list, dim=0)
        outputs_1 = torch.stack(outputs_1_list, dim=0)
        outputs = outputs_0 + outputs_1

        return outputs


class ZoneoutLSTMDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, zoneout_factor_cell=0.1, zoneout_factor_hidden=0.1, num_layers=2):
        super(ZoneoutLSTMDecoder, self).__init__()
        if zoneout_factor_cell < 0. or zoneout_factor_cell > 1. \
                or zoneout_factor_hidden < 0. or zoneout_factor_hidden > 1.:
            raise ValueError("'One/both provided Zoneout factors are not in [0, 1]")

        self.lstm_cell = nn.ModuleList()
        self.lstm_cell.append(nn.LSTMCell(input_size=input_size, hidden_size=hidden_size))
        for i in range(num_layers-1):
            self.lstm_cell.append(nn.LSTMCell(input_size=hidden_size, hidden_size=hidden_size))
        self.zoneout_cell = zoneout_factor_cell
        self.zoneout_hidden = zoneout_factor_hidden
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def forward(self, inputs, hidden_state_list_=None, cell_state_list_=None, is_training=True):
        """
        :param inputs: T(1) x B x C
        :param state:
        :param is_training:
        :return:
        """
        seq, batch_size, inputs_size= inputs.size()
        assert seq == 1
        hidden_state_list = []
        cell_state_list = []
        for i in range(self.num_layers):
            hidden_state = torch.zeros((batch_size, self.hidden_size)).cuda() \
                if hidden_state_list_ is None else hidden_state_list_[i]
            cell_state = torch.zeros((batch_size, self.hidden_size)).cuda() \
                if cell_state_list_ is None else cell_state_list_[i]
            hidden_state_list.append(hidden_state)
            cell_state_list.append(cell_state)

        output_prev_layer = inputs[0]
        for i in range(self.num_layers):
            new_hidden_state, new_cell_state = self.lstm_cell[i](output_prev_layer, (hidden_state_list[i], cell_state_list[i]))
            if is_training:
                cell_state_list[i] = (1 - self.zoneout_cell) * \
                    nn.functional.dropout(new_cell_state - cell_state_list[i], (1 - self.zoneout_cell)) + \
                    cell_state_list[i]
                hidden_state_list[i] = (1 - self.zoneout_hidden) * \
                    nn.functional.dropout(new_hidden_state - hidden_state_list[i], (1 - self.zoneout_hidden)) + \
                    hidden_state_list[i]
            else:
                cell_state_list[i] = (1 - self.zoneout_cell) * new_cell_state + self.zoneout_cell * cell_state_list[i]
                hidden_state_list[i] = (1 - self.zoneout_hidden) * new_hidden_state + self.zoneout_hidden * \
                hidden_state_list[i]
            output_prev_layer =  hidden_state_list[i]

        outputs = hidden_state_list[self.num_layers-1].unsqueeze(0)
        return outputs, (hidden_state_list, cell_state_list)  # T x B x C     # Layer x B x C

