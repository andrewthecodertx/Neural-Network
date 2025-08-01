package tui

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/charmbracelet/bubbles/textinput"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
)

// Styles
var (
	focusedStyle = lipgloss.NewStyle().Foreground(lipgloss.Color("205"))
	noStyle      = lipgloss.NewStyle()
	helpStyle    = lipgloss.NewStyle().Foreground(lipgloss.Color("241"))
)

// TrainingParams holds all the configurable values for the training session.
type TrainingParams struct {
	CsvPath           string
	HiddenLayers      []int
	HiddenActivations []string
	OutputActivation  string
	Epochs            int
	LearningRate      float64
	ErrorGoal         float64
}

// Model represents the state of the TUI.
type Model struct {
	focusIndex int
	inputs     []textinput.Model
	params     *TrainingParams
	csvFiles   []string
	quitting   bool
}

// New creates a new TUI model.
func New(csvFiles []string) Model {
	m := Model{
		inputs:   make([]textinput.Model, 7),
		params:   &TrainingParams{},
		csvFiles: csvFiles,
	}

	var t textinput.Model
	for i := range m.inputs {
		t = textinput.New()
		t.Cursor.Style = focusedStyle
		t.CharLimit = 32

		switch i {
		case 0:
			t.Placeholder = "1"
			t.Focus()
		case 1:
			t.Placeholder = "20,20"
		case 2:
			t.Placeholder = "relu,relu"
		case 3:
			t.Placeholder = "linear"
		case 4:
			t.Placeholder = "1000"
		case 5:
			t.Placeholder = "0.001"
		case 6:
			t.Placeholder = "0.001"
		}
		m.inputs[i] = t
	}

	return m
}

// Init initializes the TUI.
func (m Model) Init() tea.Cmd {
	return textinput.Blink
}

// Update handles user input and updates the model.
func (m Model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.KeyMsg:
		switch msg.String() {
		case "ctrl+c", "q":
			m.quitting = true
			return m, tea.Quit

		case "tab", "shift+tab", "enter", "up", "down":
			s := msg.String()

			if s == "enter" {
				// If it's the last field, quit
				if m.focusIndex == len(m.inputs) {
					m.quitting = true
					return m, tea.Quit
				}
				// Otherwise, move to the next field
				m.nextInput()
			}

			if s == "up" || s == "shift+tab" {
				m.prevInput()
			} else {
				m.nextInput()
			}

			// Unfocus all inputs
			for i := range m.inputs {
				m.inputs[i].Blur()
			}
			// Focus the current input
			if m.focusIndex < len(m.inputs) {
				m.inputs[m.focusIndex].Focus()
			}

			return m, nil
		}
	}

	// Handle character input
	cmd := m.updateInputs(msg)
	return m, cmd
}

func (m *Model) updateInputs(msg tea.Msg) tea.Cmd {
	var cmds []tea.Cmd
	for i := range m.inputs {
		m.inputs[i], _ = m.inputs[i].Update(msg)
		cmds = append(cmds, nil) // no-op to satisfy interface
	}
	return tea.Batch(cmds...)
}

func (m *Model) nextInput() {
	m.focusIndex = (m.focusIndex + 1) % (len(m.inputs) + 1) // +1 for the button
}

func (m *Model) prevInput() {
	m.focusIndex--
	if m.focusIndex < 0 {
		m.focusIndex = len(m.inputs)
	}
}

// View renders the TUI.
func (m Model) View() string {
	if m.quitting {
		return ""
	}

	var b strings.Builder

	b.WriteString("Neural Network Training Configuration\n\n")

	// Render CSV file list
	b.WriteString("Available CSV Files:\n")
	for i, file := range m.csvFiles {
		b.WriteString(fmt.Sprintf("  %d: %s\n", i+1, file))
	}
	b.WriteString("\n")

	// Render form
	fmt.Fprintf(&b, "Select CSV File (number): %s\n", m.inputs[0].View())
	fmt.Fprintf(&b, "Hidden Layers (e.g., 20,20): %s\n", m.inputs[1].View())
	fmt.Fprintf(&b, "Hidden Activations (e.g., relu,relu): %s\n", m.inputs[2].View())
	fmt.Fprintf(&b, "Output Activation: %s\n", m.inputs[3].View())
	fmt.Fprintf(&b, "Epochs: %s\n", m.inputs[4].View())
	fmt.Fprintf(&b, "Learning Rate: %s\n", m.inputs[5].View())
	fmt.Fprintf(&b, "Error Goal: %s\n", m.inputs[6].View())
	b.WriteString("\n")

	// Render button
	button := "[ Start Training ]"
	if m.focusIndex == len(m.inputs) {
		b.WriteString(focusedStyle.Render(button))
	} else {
		b.WriteString(button)
	}

	b.WriteString(helpStyle.Render("\n\n  ↑/↓, tab/shift+tab: navigate | enter: next | q: quit\n"))

	return b.String()
}

// Run starts the TUI and returns the configured parameters.
func (m Model) Run() (*TrainingParams, error) {
	p, err := tea.NewProgram(m).Run()
	if err != nil {
		return nil, err
	}

	finalModel := p.(Model)
	if finalModel.quitting && finalModel.focusIndex != len(finalModel.inputs) {
		return nil, fmt.Errorf("training cancelled")
	}

	// Parse values from inputs
	params := finalModel.params
	
	// CSV File
	csvIndex, _ := strconv.Atoi(finalModel.inputs[0].Value())
	if csvIndex < 1 || csvIndex > len(finalModel.csvFiles) {
		return nil, fmt.Errorf("invalid CSV file selection")
	}
	params.CsvPath = finalModel.csvFiles[csvIndex-1]

	// Hidden Layers
	layersStr := finalModel.inputs[1].Value()
	if layersStr == "" { layersStr = "20,20" }
	for _, s := range strings.Split(layersStr, ",") {
		i, err := strconv.Atoi(strings.TrimSpace(s))
		if err != nil {
			return nil, fmt.Errorf("invalid hidden layers: %w", err)
		}
		params.HiddenLayers = append(params.HiddenLayers, i)
	}

	// Hidden Activations
	activationsStr := finalModel.inputs[2].Value()
	if activationsStr == "" { activationsStr = "relu,relu" }
	params.HiddenActivations = strings.Split(activationsStr, ",")

	// Output Activation
	params.OutputActivation = finalModel.inputs[3].Value()
	if params.OutputActivation == "" { params.OutputActivation = "linear" }

	// Epochs
	epochsStr := finalModel.inputs[4].Value()
	if epochsStr == "" { epochsStr = "1000" }
	params.Epochs, _ = strconv.Atoi(epochsStr)

	// Learning Rate
	lrStr := finalModel.inputs[5].Value()
	if lrStr == "" { lrStr = "0.001" }
	params.LearningRate, _ = strconv.ParseFloat(lrStr, 64)

	// Error Goal
	egStr := finalModel.inputs[6].Value()
	if egStr == "" { egStr = "0.001" }
	params.ErrorGoal, _ = strconv.ParseFloat(egStr, 64)

	return params, nil
}
