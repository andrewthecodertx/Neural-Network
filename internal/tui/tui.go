package tui

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/charmbracelet/bubbles/textinput"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
)

// Messages
type (
	csvFilesLoadedMsg struct{ files []string }
	errCsvFiles       struct{ err error }
)

func findCsvFiles() tea.Msg {
	files, err := filepath.Glob("*.csv")
	if err != nil {
		return errCsvFiles{err}
	}
	return csvFilesLoadedMsg{files}
}

// sessionState represents the current view of the TUI.
type sessionState uint

const (
	mainMenu sessionState = iota
	trainingForm
	trainingInProgress
	predictionForm
)

// Styles
var (
	titleStyle = lipgloss.NewStyle().
			Bold(true).
			Foreground(lipgloss.Color("#FAFAFA")).
			Background(lipgloss.Color("#7D56F4")).
			PaddingLeft(1).
			PaddingRight(1)

	menuItemStyle         = lipgloss.NewStyle().PaddingLeft(2)
	selectedMenuItemStyle = lipgloss.NewStyle().PaddingLeft(2).Foreground(lipgloss.Color("205"))
	helpStyle             = lipgloss.NewStyle().Foreground(lipgloss.Color("241")).Padding(1, 0, 0, 2)
	focusedStyle          = lipgloss.NewStyle().Foreground(lipgloss.Color("205"))
)

// Model represents the state of the entire application.
type Model struct {
	state          sessionState
	menuCursor     int
	menuChoices    []string
	trainingForm   trainingFormModel
	quitting       bool
	terminalWidth  int
	terminalHeight int
}

// trainingFormModel holds the state for the training configuration form.
type trainingFormModel struct {
	focusIndex int
	inputs     []textinput.Model
	csvFiles   []string
}

func newTrainingForm() trainingFormModel {
	m := trainingFormModel{
		inputs: make([]textinput.Model, 7),
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

// New creates a new TUI model.
func New() Model {
	return Model{
		state:        mainMenu,
		menuChoices:  []string{"Train New Model", "Load Model & Predict", "Quit"},
		trainingForm: newTrainingForm(),
	}
}

// Init initializes the TUI.
func (m Model) Init() tea.Cmd {
	return textinput.Blink
}

// Update handles messages and updates the model.
func (m Model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		m.terminalWidth = msg.Width
		m.terminalHeight = msg.Height

	case csvFilesLoadedMsg:
		m.trainingForm.csvFiles = msg.files
		return m, nil

	case errCsvFiles:
		// TODO: Handle this error more gracefully
		fmt.Println("Error finding CSV files:", msg.err)
		return m, tea.Quit

	case tea.KeyMsg:
		switch m.state {
		case mainMenu:
			return m.updateMainMenu(msg)
		case trainingForm:
			return m.updateTrainingForm(msg)
		}
	}

	// Handle character input for the training form
	if m.state == trainingForm {
		cmd := m.updateTrainingInputs(msg)
		return m, cmd
	}

	return m, nil
}

func (m *Model) updateMainMenu(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	switch msg.String() {
	case "ctrl+c", "q":
		m.quitting = true
		return m, tea.Quit

	case "up", "k":
		if m.menuCursor > 0 {
			m.menuCursor--
		}

	case "down", "j":
		if m.menuCursor < len(m.menuChoices)-1 {
			m.menuCursor++
		}

	case "enter":
		switch m.menuCursor {
		case 0:
			// Transition to training form
			m.state = trainingForm
			return m, findCsvFiles
		case 1:
			// Transition to prediction form (to be implemented)
			m.state = predictionForm
		case 2:
			m.quitting = true
			return m, tea.Quit
		}
	}
	return m, nil
}

func (m Model) viewMainMenu() string {
	s := titleStyle.Render("Go Neural Network")
	s += "\n\n"

	for i, choice := range m.menuChoices {
		if m.menuCursor == i {
			s += selectedMenuItemStyle.Render(fmt.Sprintf("> %s", choice))
		} else {
			s += menuItemStyle.Render(fmt.Sprintf("  %s", choice))
		}
		s += "\n"
	}

	s += helpStyle.Render("Use arrow keys to navigate, 'enter' to select, 'q' to quit.")
	return s
}

// View renders the UI.

// View renders the UI.

func (m Model) View() string {
	if m.quitting {
		return "Quitting...\n"
	}

	var s string
	switch m.state {
	case mainMenu:
		s = m.viewMainMenu()
	case trainingForm:
		s = m.viewTrainingForm()
	// Other views will be rendered here later
	default:
		s = "Unknown state."
	}

	return s
}

func (m *Model) updateTrainingForm(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	switch msg.String() {
	case "ctrl+c", "q":
		// Go back to the main menu
		m.state = mainMenu
		return m, nil

	case "tab", "shift+tab", "enter", "up", "down":
		s := msg.String()

		// Did the user press enter while the button is focused?
		if s == "enter" && m.trainingForm.focusIndex == len(m.trainingForm.inputs) {
			// TODO: Start training
			return m, nil
		}

		// Cycle focus
		if s == "up" || s == "shift+tab" {
			m.trainingForm.focusIndex--
		} else {
			m.trainingForm.focusIndex++
		}

		if m.trainingForm.focusIndex > len(m.trainingForm.inputs) {
			m.trainingForm.focusIndex = 0
		} else if m.trainingForm.focusIndex < 0 {
			m.trainingForm.focusIndex = len(m.trainingForm.inputs)
		}

		cmds := make([]tea.Cmd, len(m.trainingForm.inputs))
		for i := 0; i <= len(m.trainingForm.inputs)-1; i++ {
			if i == m.trainingForm.focusIndex {
				// Set focused state
				cmds[i] = m.trainingForm.inputs[i].Focus()
				m.trainingForm.inputs[i].PromptStyle = focusedStyle
				m.trainingForm.inputs[i].TextStyle = focusedStyle
				continue
			}
			// Remove focused state
			m.trainingForm.inputs[i].Blur()
			m.trainingForm.inputs[i].PromptStyle = lipgloss.NewStyle()
			m.trainingForm.inputs[i].TextStyle = lipgloss.NewStyle()
		}

		return m, tea.Batch(cmds...)
	}

	// Handle character input
	cmd := m.updateTrainingInputs(msg)
	return m, cmd
}

func (m *Model) updateTrainingInputs(msg tea.Msg) tea.Cmd {
	cmds := make([]tea.Cmd, len(m.trainingForm.inputs))

	// Only update the focused input
	for i := range m.trainingForm.inputs {
		if m.trainingForm.inputs[i].Focused() {
			m.trainingForm.inputs[i], cmds[i] = m.trainingForm.inputs[i].Update(msg)
		}
	}

	return tea.Batch(cmds...)
}

func (m Model) viewTrainingForm() string {
	var b strings.Builder

	b.WriteString("Neural Network Training Configuration\n\n")

	// Render CSV file list
	b.WriteString("Available CSV Files:\n")
	if len(m.trainingForm.csvFiles) == 0 {
		b.WriteString("  (No CSV files found in current directory)\n")
	} else {
		for i, file := range m.trainingForm.csvFiles {
			b.WriteString(fmt.Sprintf("  %d: %s\n", i+1, file))
		}
	}
	b.WriteString("\n")

	// Render form
	fmt.Fprintf(&b, "Select CSV File (number): %s\n", m.trainingForm.inputs[0].View())
	fmt.Fprintf(&b, "Hidden Layers (e.g., 20,20): %s\n", m.trainingForm.inputs[1].View())
	fmt.Fprintf(&b, "Hidden Activations (e.g., relu,relu): %s\n", m.trainingForm.inputs[2].View())
	fmt.Fprintf(&b, "Output Activation: %s\n", m.trainingForm.inputs[3].View())
	fmt.Fprintf(&b, "Epochs: %s\n", m.trainingForm.inputs[4].View())
	fmt.Fprintf(&b, "Learning Rate: %s\n", m.trainingForm.inputs[5].View())
	fmt.Fprintf(&b, "Error Goal: %s\n", m.trainingForm.inputs[6].View())
	b.WriteString("\n")

	// Render button
	button := "[ Start Training ]"
	if m.trainingForm.focusIndex == len(m.trainingForm.inputs) {
		b.WriteString(focusedStyle.Render(button))
	} else {
		b.WriteString(button)
	}

	b.WriteString(helpStyle.Render("\n\n  ↑/↓, tab/shift+tab: navigate | enter: select | q: back\n"))

	return b.String()
}

// Start begins the TUI application.
func Start() {
	p := tea.NewProgram(New(), tea.WithAltScreen())
	if _, err := p.Run(); err != nil {
		fmt.Printf("Alas, there's been an error: %v", err)
		os.Exit(1)
	}
}

// Start begins the TUI application.
func Start() {
	p := tea.NewProgram(New(), tea.WithAltScreen())
	if _, err := p.Run(); err != nil {
		fmt.Printf("Alas, there's been an error: %v", err)
		os.Exit(1)
	}
}


// Start begins the TUI application.
func Start() {
	p := tea.NewProgram(New(), tea.WithAltScreen())
	if _, err := p.Run(); err != nil {
		fmt.Printf("Alas, there's been an error: %v", err)
		os.Exit(1)
	}
}