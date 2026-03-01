"""
╔══════════════════════════════════════════════════════════════╗
║  AI Insights Engine — Generates intelligent insights,       ║
║  health scores, recommendations & predictive alerts         ║
╚══════════════════════════════════════════════════════════════╝

Provides:
    1. System Health Score (0-100)
    2. Smart Recommendations
    3. Efficiency Analysis
    4. Predictive Alerts
    5. Energy Optimization Tips
    6. Appliance Usage Patterns
    7. Cost Optimization Insights
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque
from datetime import datetime, timedelta

from .config import (
    VOLTAGE_NOMINAL, VOLTAGE_HIGH, VOLTAGE_LOW,
    CURRENT_MAX, POWER_MAX, COST_PER_KWH, CURRENCY,
    HEALTH_SCORE_WEIGHTS, FAULT_CLASSES, FAULT_SEVERITY
)


class InsightsEngine:
    """
    AI-powered insights generator for the power monitoring system.
    
    Combines ML model outputs with domain expertise to generate
    actionable insights, health scores, and recommendations.
    
    Usage:
        engine = InsightsEngine()
        engine.update(voltage, current, power, anomaly_result, fault_result)
        
        health = engine.get_health_score()
        insights = engine.get_insights()
        recommendations = engine.get_recommendations()
    """

    def __init__(self):
        # Data buffers
        self.voltage_history = deque(maxlen=3600)   # ~1 hour at 1 Hz
        self.current_history = deque(maxlen=3600)
        self.power_history = deque(maxlen=3600)
        self.energy_log = deque(maxlen=86400)        # ~24 hours
        
        # ML results buffers
        self.anomaly_scores = deque(maxlen=1000)
        self.fault_predictions = deque(maxlen=1000)
        
        # Session tracking
        self.session_start = datetime.now()
        self.total_energy_kwh = 0.0
        self.last_update_time = datetime.now()
        self.update_count = 0
        
        # Alert tracking
        self.active_alerts = []
        self.alert_history = deque(maxlen=500)
        
        # Insight cache
        self._cached_insights = {}
        self._cache_age = 0

    def update(self, voltage: float, current: float, power: float,
               anomaly_result: Optional[Dict] = None,
               fault_result: Optional[Dict] = None):
        """
        Update the engine with new sensor readings and ML results.
        
        Args:
            voltage, current, power: Current sensor values
            anomaly_result: Dict with 'is_anomaly', 'score', 'details'
            fault_result: Dict with 'fault_id', 'confidence', 'top3'
        """
        now = datetime.now()
        
        # Store sensor data
        self.voltage_history.append(voltage)
        self.current_history.append(current)
        self.power_history.append(power)
        
        # Update energy
        dt_hours = (now - self.last_update_time).total_seconds() / 3600.0
        self.total_energy_kwh += (power / 1000.0) * dt_hours
        self.last_update_time = now
        self.update_count += 1
        
        # Store energy snapshot
        self.energy_log.append({
            'time': now,
            'power': power,
            'energy_cumulative': self.total_energy_kwh,
        })
        
        # Store ML results
        if anomaly_result:
            self.anomaly_scores.append(anomaly_result.get('score', 0))
        
        if fault_result:
            self.fault_predictions.append({
                'fault_id': fault_result.get('fault_id', 0),
                'confidence': fault_result.get('confidence', 0),
                'time': now,
            })
        
        # Check for alerts
        self._check_alerts(voltage, current, power, anomaly_result, fault_result)

    def get_health_score(self) -> Dict:
        """
        Calculate overall system health score (0-100).
        
        Components:
            - Voltage Stability (25%)
            - Current Safety (20%)
            - Power Efficiency (20%)
            - Anomaly Rate (15%)
            - Power Factor (10%)
            - Energy Trend (10%)
        """
        scores = {}
        
        # ── Voltage Stability (0-100) ──
        if len(self.voltage_history) > 10:
            v = np.array(self.voltage_history)
            v_std = np.std(v)
            v_mean = np.mean(v)
            deviation = abs(v_mean - VOLTAGE_NOMINAL) / VOLTAGE_NOMINAL
            
            stability = max(0, 100 - (v_std * 10) - (deviation * 200))
            scores['voltage_stability'] = min(100, stability)
        else:
            scores['voltage_stability'] = 50  # Unknown
        
        # ── Current Safety (0-100) ──
        if len(self.current_history) > 10:
            i = np.array(self.current_history)
            max_current = np.max(i)
            usage_pct = max_current / CURRENT_MAX
            
            if usage_pct <= 0.7:
                scores['current_safety'] = 100
            elif usage_pct <= 0.85:
                scores['current_safety'] = 100 - (usage_pct - 0.7) * 200
            elif usage_pct <= 1.0:
                scores['current_safety'] = max(0, 70 - (usage_pct - 0.85) * 400)
            else:
                scores['current_safety'] = max(0, 10 - (usage_pct - 1.0) * 100)
        else:
            scores['current_safety'] = 50
        
        # ── Power Efficiency (0-100) ──
        if len(self.power_history) > 10:
            p = np.array(self.power_history)
            load_pct = np.mean(p) / POWER_MAX
            
            if load_pct < 0.8:
                scores['power_efficiency'] = 100 - (load_pct * 30)
            else:
                scores['power_efficiency'] = max(0, 100 - (load_pct * 80))
        else:
            scores['power_efficiency'] = 50
        
        # ── Anomaly Rate (0-100) ──
        if len(self.anomaly_scores) > 10:
            anomaly_rate = np.mean([1 if s > 0.3 else 0 for s in self.anomaly_scores])
            scores['anomaly_rate'] = max(0, 100 - anomaly_rate * 200)
        else:
            scores['anomaly_rate'] = 80  # Assume OK
        
        # ── Power Factor (0-100) ──
        if len(self.voltage_history) > 10 and len(self.current_history) > 10:
            v = np.array(self.voltage_history)
            i = np.array(self.current_history)
            p = np.array(self.power_history)
            
            apparent = np.mean(v) * np.mean(i)
            pf = np.mean(p) / apparent if apparent > 0 else 0
            
            if pf >= 0.95:
                scores['power_factor'] = 100
            elif pf >= 0.85:
                scores['power_factor'] = 80 + (pf - 0.85) * 200
            elif pf >= 0.7:
                scores['power_factor'] = 40 + (pf - 0.7) * 267
            else:
                scores['power_factor'] = max(0, pf * 57)
        else:
            scores['power_factor'] = 50
        
        # ── Energy Trend (0-100) ──
        if len(self.energy_log) > 60:
            recent_rates = [e['power'] for e in list(self.energy_log)[-60:]]
            trend_slope = np.polyfit(range(len(recent_rates)), recent_rates, 1)[0]
            
            if abs(trend_slope) < 1:
                scores['energy_trend'] = 90  # Stable
            elif trend_slope < 0:
                scores['energy_trend'] = 95  # Decreasing — good
            else:
                scores['energy_trend'] = max(0, 80 - trend_slope * 5)
        else:
            scores['energy_trend'] = 70
        
        # ── Weighted composite score ──
        total_score = 0
        for component, weight in HEALTH_SCORE_WEIGHTS.items():
            total_score += scores.get(component, 50) * weight
        
        # Determine health label
        if total_score >= 90:
            label = "Excellent"
            color = "#22c55e"
        elif total_score >= 75:
            label = "Good"
            color = "#34d399"
        elif total_score >= 60:
            label = "Fair"
            color = "#eab308"
        elif total_score >= 40:
            label = "Poor"
            color = "#f97316"
        else:
            label = "Critical"
            color = "#ef4444"
        
        return {
            'overall_score': round(total_score, 1),
            'label': label,
            'color': color,
            'components': scores,
            'weights': HEALTH_SCORE_WEIGHTS,
        }

    def get_insights(self) -> List[Dict]:
        """
        Generate AI-driven insights based on current system state.
        
        Returns list of insight dicts with:
            - category, title, description, severity, icon
        """
        insights = []
        
        if self.update_count < 10:
            return [{'category': 'info', 'title': 'Collecting Data',
                     'description': 'Gathering sensor data for analysis. Insights will appear shortly.',
                     'severity': 'info', 'icon': '📡'}]
        
        # ── Voltage Analysis ──
        if len(self.voltage_history) > 20:
            v = np.array(self.voltage_history)
            v_mean = np.mean(v)
            v_std = np.std(v)
            
            if v_mean > VOLTAGE_HIGH:
                insights.append({
                    'category': 'voltage',
                    'title': 'High Voltage Detected',
                    'description': f'Average voltage ({v_mean:.1f}V) exceeds safe limit ({VOLTAGE_HIGH}V). '
                                   'This may damage sensitive electronics. Consider a voltage stabilizer.',
                    'severity': 'danger',
                    'icon': '⚡',
                })
            elif v_mean < VOLTAGE_LOW:
                insights.append({
                    'category': 'voltage',
                    'title': 'Low Voltage Alert',
                    'description': f'Average voltage ({v_mean:.1f}V) is below minimum ({VOLTAGE_LOW}V). '
                                   'Motors and compressors may overheat. Contact your utility provider.',
                    'severity': 'danger',
                    'icon': '🔋',
                })
            elif v_std > 5:
                insights.append({
                    'category': 'voltage',
                    'title': 'Voltage Instability',
                    'description': f'Voltage fluctuation is high (σ={v_std:.1f}V). '
                                   'This could indicate grid instability or heavy nearby loads.',
                    'severity': 'warning',
                    'icon': '📉',
                })
            else:
                insights.append({
                    'category': 'voltage',
                    'title': 'Voltage Stable',
                    'description': f'Voltage is stable at {v_mean:.1f}V (nominal: {VOLTAGE_NOMINAL}V). '
                                   'Grid quality is good.',
                    'severity': 'success',
                    'icon': '✅',
                })
        
        # ── Current Analysis ──
        if len(self.current_history) > 20:
            i = np.array(self.current_history)
            i_mean = np.mean(i)
            i_max = np.max(i)
            
            if i_max > CURRENT_MAX:
                insights.append({
                    'category': 'current',
                    'title': 'Overcurrent Detected',
                    'description': f'Peak current ({i_max:.2f}A) exceeds limit ({CURRENT_MAX}A). '
                                   'Check for short circuits or overloaded circuits. Risk of fire!',
                    'severity': 'danger',
                    'icon': '🔥',
                })
            elif i_max > CURRENT_MAX * 0.85:
                insights.append({
                    'category': 'current',
                    'title': 'Current Near Limit',
                    'description': f'Peak current ({i_max:.2f}A) is {i_max/CURRENT_MAX*100:.0f}% of limit. '
                                   'Consider redistributing loads across circuits.',
                    'severity': 'warning',
                    'icon': '⚠️',
                })
        
        # ── Power & Load Analysis ──
        if len(self.power_history) > 20:
            p = np.array(self.power_history)
            p_mean = np.mean(p)
            p_max = np.max(p)
            
            load_pct = p_mean / POWER_MAX * 100
            
            if p_max > POWER_MAX:
                insights.append({
                    'category': 'power',
                    'title': 'Power Overload!',
                    'description': f'Peak power ({p_max:.0f}W) exceeds rated capacity ({POWER_MAX}W). '
                                   'Reduce active loads immediately to prevent breaker trip.',
                    'severity': 'danger',
                    'icon': '💥',
                })
            elif load_pct > 70:
                insights.append({
                    'category': 'power',
                    'title': 'High Power Usage',
                    'description': f'Average load is {load_pct:.0f}% of capacity ({p_mean:.0f}W / {POWER_MAX}W). '
                                   f'Estimated cost: {CURRENCY}{self.total_energy_kwh * COST_PER_KWH:.2f}',
                    'severity': 'warning',
                    'icon': '📊',
                })
            else:
                insights.append({
                    'category': 'power',
                    'title': 'Normal Power Usage',
                    'description': f'Load at {load_pct:.0f}% capacity ({p_mean:.0f}W). '
                                   f'Running efficiently.',
                    'severity': 'success',
                    'icon': '💡',
                })
        
        # ── Power Factor Analysis ──
        if len(self.voltage_history) > 20 and len(self.current_history) > 20:
            v_mean = np.mean(list(self.voltage_history))
            i_mean = np.mean(list(self.current_history))
            p_mean = np.mean(list(self.power_history))
            
            apparent = v_mean * i_mean
            pf = p_mean / apparent if apparent > 0 else 0
            
            if pf < 0.7:
                insights.append({
                    'category': 'efficiency',
                    'title': 'Poor Power Factor',
                    'description': f'Power factor is {pf:.3f} (ideal: >0.95). '
                                   'You\'re paying for wasted reactive power. '
                                   'Consider power factor correction capacitors.',
                    'severity': 'warning',
                    'icon': '📉',
                })
            elif pf < 0.85:
                insights.append({
                    'category': 'efficiency',
                    'title': 'Moderate Power Factor',
                    'description': f'Power factor is {pf:.3f}. '
                                   'Some reactive power waste detected. Inductive loads (motors, transformers) '
                                   'may benefit from PF correction.',
                    'severity': 'info',
                    'icon': '📊',
                })
        
        # ── Anomaly Insights ──
        if len(self.anomaly_scores) > 50:
            anomaly_rate = np.mean([1 if s > 0.3 else 0 for s in self.anomaly_scores])
            if anomaly_rate > 0.1:
                insights.append({
                    'category': 'anomaly',
                    'title': 'Frequent Anomalies Detected',
                    'description': f'{anomaly_rate*100:.1f}% of recent readings are anomalous. '
                                   'The system may need maintenance or inspection.',
                    'severity': 'warning',
                    'icon': '🔍',
                })
        
        # ── Fault Pattern ──
        if len(self.fault_predictions) > 30:
            recent_faults = list(self.fault_predictions)[-30:]
            non_normal = [f for f in recent_faults if f['fault_id'] != 0]
            if len(non_normal) > 5:
                fault_names = [FAULT_CLASSES.get(f['fault_id'], 'Unknown') for f in non_normal]
                most_common = max(set(fault_names), key=fault_names.count)
                insights.append({
                    'category': 'fault',
                    'title': f'Recurring Fault: {most_common}',
                    'description': f'"{most_common}" has been detected {fault_names.count(most_common)} times '
                                   f'in the last 30 readings. Investigate the root cause.',
                    'severity': 'warning',
                    'icon': '🛠️',
                })
        
        # ── Energy Cost Insight ──
        if self.total_energy_kwh > 0:
            cost = self.total_energy_kwh * COST_PER_KWH
            session_hours = (datetime.now() - self.session_start).total_seconds() / 3600
            if session_hours > 0:
                hourly_cost = cost / session_hours
                daily_projected = hourly_cost * 24
                monthly_projected = daily_projected * 30
                
                insights.append({
                    'category': 'cost',
                    'title': 'Energy Cost Analysis',
                    'description': (
                        f'Session energy: {self.total_energy_kwh:.4f} kWh '
                        f'({CURRENCY}{cost:.2f})\n'
                        f'Projected daily: {CURRENCY}{daily_projected:.2f} | '
                        f'Monthly: {CURRENCY}{monthly_projected:.2f}'
                    ),
                    'severity': 'info',
                    'icon': '💰',
                })
        
        return insights

    def get_recommendations(self) -> List[Dict]:
        """
        Generate actionable recommendations based on analysis.
        """
        recs = []
        
        if self.update_count < 30:
            return [{'priority': 'low', 'title': 'Gathering Data',
                     'action': 'Continue monitoring. Recommendations will appear after ~30 seconds.',
                     'category': 'info'}]
        
        # Voltage recommendations
        if len(self.voltage_history) > 30:
            v = np.array(self.voltage_history)
            v_mean = np.mean(v)
            v_std = np.std(v)
            
            if v_mean > VOLTAGE_HIGH or v_mean < VOLTAGE_LOW:
                recs.append({
                    'priority': 'high',
                    'title': 'Install Voltage Stabilizer',
                    'action': f'Voltage is {v_mean:.1f}V (acceptable: {VOLTAGE_LOW}-{VOLTAGE_HIGH}V). '
                              'Install an automatic voltage regulator (AVR) to protect equipment.',
                    'category': 'hardware',
                    'estimated_savings': f'{CURRENCY}500-2000 in prevented damage',
                })
            
            if v_std > 8:
                recs.append({
                    'priority': 'medium',
                    'title': 'Investigate Voltage Fluctuations',
                    'action': 'High voltage variance detected. Check for loose connections, '
                              'nearby heavy industrial loads, or contact utility for grid assessment.',
                    'category': 'maintenance',
                })
        
        # Load balancing recommendations
        if len(self.power_history) > 60:
            p = np.array(self.power_history)
            p_mean = np.mean(p)
            p_std = np.std(p)
            
            if p_mean / POWER_MAX > 0.7:
                recs.append({
                    'priority': 'high',
                    'title': 'Reduce Peak Load',
                    'action': 'You\'re using >70% of circuit capacity. Redistribute high-power '
                              'appliances (AC, heater, iron) across different circuits or time slots.',
                    'category': 'optimization',
                })
            
            if p_std > p_mean * 0.5:
                recs.append({
                    'priority': 'medium',
                    'title': 'Stabilize Load Profile',
                    'action': 'Power consumption is highly variable. Consider using timers or smart '
                              'plugs to stagger high-power appliances and flatten the load curve.',
                    'category': 'optimization',
                })
        
        # Power factor recommendations
        if len(self.voltage_history) > 30 and len(self.current_history) > 30:
            v_mean = np.mean(list(self.voltage_history))
            i_mean = np.mean(list(self.current_history))
            p_mean = np.mean(list(self.power_history))
            apparent = v_mean * i_mean
            pf = p_mean / apparent if apparent > 0 else 1
            
            if pf < 0.85:
                wasted_pct = (1 - pf) * 100
                recs.append({
                    'priority': 'medium',
                    'title': 'Improve Power Factor',
                    'action': f'Power factor is {pf:.3f} — {wasted_pct:.0f}% of electrical capacity is wasted. '
                              'Add power factor correction capacitors near inductive loads (motors, pumps).',
                    'category': 'efficiency',
                    'estimated_savings': f'~{wasted_pct:.0f}% reduction in apparent power demand',
                })
        
        # Energy saving tips
        if self.total_energy_kwh > 0.01:
            recs.append({
                'priority': 'low',
                'title': 'Energy Saving Tips',
                'action': '• Use LED lighting (saves ~75% vs incandescent)\n'
                          '• Set AC to 24°C (each degree lower adds ~6% cost)\n'
                          '• Use star-rated appliances\n'
                          '• Schedule heavy loads during off-peak hours\n'
                          '• Unplug phantom loads (chargers, standby devices)',
                'category': 'savings',
            })
        
        # Sort by priority
        priority_order = {'high': 0, 'medium': 1, 'low': 2}
        recs.sort(key=lambda x: priority_order.get(x['priority'], 3))
        
        return recs

    def _check_alerts(self, v: float, i: float, p: float,
                      anomaly_result: Optional[Dict],
                      fault_result: Optional[Dict]):
        """Check and generate real-time alerts."""
        now = datetime.now()
        new_alerts = []
        
        # Threshold alerts
        if v > VOLTAGE_HIGH:
            new_alerts.append({
                'type': 'overvoltage',
                'message': f'⚡ OVERVOLTAGE: {v:.1f}V (limit: {VOLTAGE_HIGH}V)',
                'severity': 'danger',
                'time': now,
            })
        
        if v < VOLTAGE_LOW:
            new_alerts.append({
                'type': 'undervoltage',
                'message': f'🔋 UNDERVOLTAGE: {v:.1f}V (min: {VOLTAGE_LOW}V)',
                'severity': 'danger',
                'time': now,
            })
        
        if i > CURRENT_MAX:
            new_alerts.append({
                'type': 'overcurrent',
                'message': f'🔥 OVERCURRENT: {i:.2f}A (limit: {CURRENT_MAX}A)',
                'severity': 'danger',
                'time': now,
            })
        
        if p > POWER_MAX:
            new_alerts.append({
                'type': 'overload',
                'message': f'💥 OVERLOAD: {p:.0f}W (limit: {POWER_MAX}W)',
                'severity': 'danger',
                'time': now,
            })
        
        # ML-based alerts
        if anomaly_result and anomaly_result.get('score', 0) > 0.7:
            new_alerts.append({
                'type': 'anomaly',
                'message': f'🔍 ANOMALY DETECTED (score: {anomaly_result["score"]:.2f})',
                'severity': 'warning',
                'time': now,
            })
        
        if fault_result and fault_result.get('fault_id', 0) != 0:
            fault_name = FAULT_CLASSES.get(fault_result['fault_id'], 'Unknown')
            confidence = fault_result.get('confidence', 0)
            if confidence > 0.7:
                new_alerts.append({
                    'type': 'fault',
                    'message': f'🛠️ FAULT: {fault_name} (confidence: {confidence:.0%})',
                    'severity': 'warning',
                    'time': now,
                })
        
        # Update active alerts
        self.active_alerts = new_alerts
        for alert in new_alerts:
            self.alert_history.append(alert)

    def get_active_alerts(self) -> List[Dict]:
        """Get currently active alerts."""
        return self.active_alerts.copy()

    def get_alert_summary(self, last_n: int = 100) -> Dict:
        """Get summary of recent alerts."""
        recent = list(self.alert_history)[-last_n:]
        if not recent:
            return {'total': 0, 'by_type': {}, 'by_severity': {}}
        
        by_type = {}
        by_severity = {}
        for alert in recent:
            t = alert['type']
            s = alert['severity']
            by_type[t] = by_type.get(t, 0) + 1
            by_severity[s] = by_severity.get(s, 0) + 1
        
        return {
            'total': len(recent),
            'by_type': by_type,
            'by_severity': by_severity,
        }

    def get_efficiency_report(self) -> Dict:
        """Generate a comprehensive efficiency report."""
        if self.update_count < 60:
            return {'status': 'insufficient_data', 'message': 'Need at least 60 samples'}
        
        v = np.array(self.voltage_history)
        i = np.array(self.current_history)
        p = np.array(self.power_history)
        
        apparent = np.mean(v) * np.mean(i)
        pf = np.mean(p) / apparent if apparent > 0 else 0
        
        session_hours = (datetime.now() - self.session_start).total_seconds() / 3600
        avg_power = np.mean(p)
        
        return {
            'status': 'ready',
            'session_duration_hours': round(session_hours, 2),
            'total_energy_kwh': round(self.total_energy_kwh, 4),
            'total_cost': round(self.total_energy_kwh * COST_PER_KWH, 2),
            'avg_power_w': round(avg_power, 1),
            'peak_power_w': round(np.max(p), 1),
            'min_power_w': round(np.min(p), 1),
            'power_factor': round(pf, 3),
            'load_utilization_pct': round(avg_power / POWER_MAX * 100, 1),
            'voltage_stability': round(100 - np.std(v) * 10, 1),
            'projected_daily_kwh': round(avg_power * 24 / 1000, 2),
            'projected_monthly_cost': round(avg_power * 24 * 30 / 1000 * COST_PER_KWH, 2),
            'currency': CURRENCY,
        }
