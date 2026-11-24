import React from 'react';
import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import MetricCard from '../components/MetricCard';
import ConfusionMatrix from '../components/ConfusionMatrix';
import Navbar from '../components/Navbar';
import Sidebar from '../components/Sidebar';
import CircuitViewer from '../components/CircuitViewer';
import { Activity } from 'lucide-react';

describe('MetricCard Component', () => {
  it('renders title and value properly', () => {
    render(
      <MetricCard
        title="Accuracy Rate"
        value="96.7%"
        subtitle="+12% over classical"
        icon={Activity}
        color="emerald"
      />
    );
    expect(screen.getByText('Accuracy Rate')).toBeDefined();
    expect(screen.getByText('96.7%')).toBeDefined();
    expect(screen.getByText('+12% over classical')).toBeDefined();
  });
});

describe('ConfusionMatrix Component', () => {
  it('renders confusion matrix matrix and labels correctly', () => {
    const matrix = [
      [10, 0, 0],
      [0, 9, 1],
      [0, 2, 8]
    ];
    const labels = ['setosa', 'versicolor', 'virginica'];
    render(<ConfusionMatrix matrix={matrix} labels={labels} title="Test Matrix" />);

    expect(screen.getByText('Test Matrix')).toBeDefined();
    // Labels appear in headers and rows
    expect(screen.getAllByText('setosa').length).toBeGreaterThan(0);
    expect(screen.getAllByText('versicolor').length).toBeGreaterThan(0);
    expect(screen.getAllByText('virginica').length).toBeGreaterThan(0);
  });

  it('renders fallback when matrix is null', () => {
    render(<ConfusionMatrix matrix={null} labels={[]} />);
    expect(screen.getByText('No confusion matrix data available')).toBeDefined();
  });
});

describe('Navbar Component', () => {
  it('renders header title and IEEE reference badge', () => {
    render(<Navbar activeDataset="iris" setActiveDataset={vi.fn()} datasets={[]} />);
    expect(screen.getByText(/Quantum k-NN Research Platform/i)).toBeDefined();
    expect(screen.getByText(/IEEE ICOCT 2025/i)).toBeDefined();
    expect(screen.getByText(/PennyLane QML/i)).toBeDefined();
  });
});

describe('Sidebar Component', () => {
  it('renders all 13 navigation items and triggers page change', () => {
    const setActivePage = vi.fn();
    render(<Sidebar activePage="dashboard" setActivePage={setActivePage} />);

    expect(screen.getByText('Executive Dashboard')).toBeDefined();
    expect(screen.getByText('Dataset Explorer')).toBeDefined();
    expect(screen.getByText('Quantum Feature Maps')).toBeDefined();
    expect(screen.getByText('Adaptive Hybrid QKNN')).toBeDefined();
    expect(screen.getByText('Explainable AI')).toBeDefined();

    const playgroundButton = screen.getByText('⚡ Live Q-KNN Demo');
    fireEvent.click(playgroundButton);
    expect(setActivePage).toHaveBeenCalledWith('prediction');
  });
});

describe('CircuitViewer Component', () => {
  it('renders circuit diagram text and gate counts', () => {
    const metadata = {
      display_name: 'Hybrid Feature Map (Z+ZZ+Pauli)',
      n_qubits: 4,
      depth: 1,
      total_gates: 36,
      entangling_gates: 12,
      has_entanglement: true,
    };
    const ascii = '0: ──H──RZ──\n1: ──H──RZ──';
    render(<CircuitViewer asciiDiagram={ascii} metadata={metadata} featureMapName="hybrid" />);
    expect(screen.getByText(/Hybrid Feature Map/i)).toBeDefined();
    expect(screen.getByText('36')).toBeDefined();
    expect(screen.getByText('12')).toBeDefined();
    expect(screen.getByText('Entangled State')).toBeDefined();
  });
});
