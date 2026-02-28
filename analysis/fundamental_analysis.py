"""
Fundamental Analysis Engine for Indian Stock Market

Features:
- Valuation Analysis (PE, PB, EV/EBITDA, DCF)
- Profitability Analysis (ROE, ROA, Margins)
- Growth Analysis (Revenue, EPS, Book Value)
- Financial Health (Debt ratios, Interest coverage)
- Efficiency Ratios (Asset turnover, Inventory)
- DuPont Analysis
- Piotroski F-Score
- Altman Z-Score
- Graham Number
- PEG Ratio Analysis
- Intrinsic Value Calculation (DCF)
- Peer Comparison
- Sector-relative Valuation
- Quality Score (composite)
- Dividend Analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


# ============================================================
# DATA STRUCTURES
# ============================================================

class FundamentalRating(Enum):
    """Stock fundamental rating"""
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


@dataclass
class ValuationResult:
    """Valuation analysis result"""
    symbol: str
    current_price: float
    intrinsic_value: float
    graham_number: float
    margin_of_safety: float  # %
    pe_ratio: float
    pb_ratio: float
    peg_ratio: float
    ev_ebitda: float
    ps_ratio: float
    valuation_rating: str  # undervalued, fair, overvalued
    details: Dict = field(default_factory=dict)


@dataclass
class QualityScore:
    """Composite quality score"""
    symbol: str
    total_score: float  # 0-100
    profitability_score: float
    growth_score: float
    financial_health_score: float
    valuation_score: float
    efficiency_score: float
    piotroski_score: int  # 0-9
    altman_z_score: float
    rating: FundamentalRating
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)


# ============================================================
# VALUATION ANALYSIS
# ============================================================

class ValuationAnalyzer:
    """Stock valuation using multiple methods"""

    @staticmethod
    def graham_number(eps: float, book_value: float) -> float:
        """
        Benjamin Graham's intrinsic value formula
        
        Graham Number = âˆš(22.5 Ã— EPS Ã— Book Value)
        
        If price < Graham Number â†’ Undervalued
        """
        if eps <= 0 or book_value <= 0:
            return 0
        return np.sqrt(22.5 * eps * book_value)

    @staticmethod
    def graham_formula(
        eps: float,
        growth_rate: float,
        aaa_yield: float = 7.0
    ) -> float:
        """
        Graham's growth formula
        
        V = EPS Ã— (8.5 + 2g) Ã— 4.4 / Y
        
        Where:
            g = expected growth rate (%)
            Y = current AAA corporate bond yield (%)
            8.5 = base PE for no-growth company
        """
        if eps <= 0:
            return 0
        return eps * (8.5 + 2 * growth_rate) * 4.4 / aaa_yield

    @staticmethod
    def dcf_valuation(
        free_cash_flows: List[float],
        growth_rate: float = 0.10,
        terminal_growth: float = 0.03,
        discount_rate: float = 0.12,
        shares_outstanding: int = 1,
        projection_years: int = 10
    ) -> Dict:
        """
        Discounted Cash Flow (DCF) Valuation
        
        Args:
            free_cash_flows: Historical FCF (latest first)
            growth_rate: Expected FCF growth rate
            terminal_growth: Long-term growth rate (usually GDP growth)
            discount_rate: WACC or required rate of return
            shares_outstanding: Total shares
            projection_years: Years to project
        
        Returns:
            Dictionary with DCF valuation details
        """
        if not free_cash_flows or free_cash_flows[0] <= 0:
            return {'intrinsic_value_per_share': 0, 'error': 'Invalid FCF'}

        base_fcf = free_cash_flows[0]

        # Project future cash flows
        projected_fcf = []
        for year in range(1, projection_years + 1):
            fcf = base_fcf * (1 + growth_rate) ** year
            projected_fcf.append(fcf)

        # Terminal value
        terminal_fcf = projected_fcf[-1] * (1 + terminal_growth)
        terminal_value = terminal_fcf / (discount_rate - terminal_growth)

        # Discount all cash flows
        total_pv = 0
        for year, fcf in enumerate(projected_fcf, 1):
            pv = fcf / (1 + discount_rate) ** year
            total_pv += pv

        # Discount terminal value
        terminal_pv = terminal_value / (1 + discount_rate) ** projection_years
        total_pv += terminal_pv

        intrinsic_value = total_pv / shares_outstanding if shares_outstanding > 0 else 0

        return {
            'intrinsic_value_per_share': intrinsic_value,
            'total_present_value': total_pv,
            'terminal_value': terminal_value,
            'terminal_pv': terminal_pv,
            'projected_fcf': projected_fcf,
            'base_fcf': base_fcf,
            'growth_rate': growth_rate,
            'discount_rate': discount_rate,
        }

    @staticmethod
    def relative_valuation(
        stock_metrics: Dict,
        sector_avg: Dict
    ) -> Dict:
        """
        Relative valuation compared to sector averages
        
        Args:
            stock_metrics: {'pe': x, 'pb': y, 'ev_ebitda': z, ...}
            sector_avg: Same structure with sector averages
        """
        result = {}

        metrics = ['pe', 'pb', 'ev_ebitda', 'ps', 'dividend_yield']

        for metric in metrics:
            stock_val = stock_metrics.get(metric, 0)
            sector_val = sector_avg.get(metric, 0)

            if sector_val > 0 and stock_val > 0:
                ratio = stock_val / sector_val
                result[f'{metric}_vs_sector'] = ratio
                result[f'{metric}_premium_discount'] = (ratio - 1) * 100

                if metric in ['pe', 'pb', 'ev_ebitda', 'ps']:
                    # Lower is better for these
                    result[f'{metric}_signal'] = (
                        'undervalued' if ratio < 0.8 else
                        'fair' if ratio < 1.2 else
                        'overvalued'
                    )
                else:
                    # Higher is better for dividend yield
                    result[f'{metric}_signal'] = (
                        'attractive' if ratio > 1.2 else
                        'fair' if ratio > 0.8 else
                        'below_average'
                    )

        return result

    @staticmethod
    def margin_of_safety(
        current_price: float,
        intrinsic_value: float
    ) -> float:
        """
        Margin of Safety = (Intrinsic Value - Price) / Intrinsic Value Ã— 100
        
        > 25% = Good margin of safety
        > 50% = Excellent margin of safety
        """
        if intrinsic_value <= 0:
            return -100
        return ((intrinsic_value - current_price) / intrinsic_value) * 100

    @staticmethod
    def peg_ratio(pe_ratio: float, earnings_growth: float) -> float:
        """
        PEG Ratio = PE / Earnings Growth Rate
        
        < 1.0 = Potentially undervalued
        1.0 = Fairly valued
        > 1.0 = Potentially overvalued
        """
        if earnings_growth <= 0:
            return float('inf')
        return pe_ratio / (earnings_growth * 100)

    @staticmethod
    def earnings_yield(eps: float, price: float) -> float:
        """
        Earnings Yield = EPS / Price Ã— 100
        
        Compare with bond yields to assess relative value
        """
        if price <= 0:
            return 0
        return (eps / price) * 100


# ============================================================
# PROFITABILITY ANALYSIS
# ============================================================

class ProfitabilityAnalyzer:
    """Analyze company profitability"""

    @staticmethod
    def dupont_analysis(
        net_income: float,
        revenue: float,
        total_assets: float,
        total_equity: float
    ) -> Dict:
        """
        DuPont Analysis - Decompose ROE
        
        ROE = Net Margin Ã— Asset Turnover Ã— Equity Multiplier
        
        Helps identify SOURCE of profitability:
        - High margin business? (TCS, HINDUNILVR)
        - High asset turnover? (retail, FMCG)
        - High leverage? (banks, NBFCs)
        """
        if revenue <= 0 or total_assets <= 0 or total_equity <= 0:
            return {}

        net_margin = net_income / revenue
        asset_turnover = revenue / total_assets
        equity_multiplier = total_assets / total_equity

        roe = net_margin * asset_turnover * equity_multiplier

        # Classify the profitability driver
        driver = 'margin' if net_margin > 0.15 else (
            'turnover' if asset_turnover > 1.5 else 'leverage'
        )

        return {
            'roe': roe,
            'net_margin': net_margin,
            'asset_turnover': asset_turnover,
            'equity_multiplier': equity_multiplier,
            'primary_driver': driver,
            'margin_contribution': net_margin * 100,
            'turnover_contribution': asset_turnover,
            'leverage_contribution': equity_multiplier,
        }

    @staticmethod
    def profitability_score(metrics: Dict) -> float:
        """
        Calculate profitability score (0-100)
        
        Args:
            metrics: Dictionary with profitability metrics
        """
        score = 0
        max_score = 0

        # ROE
        roe = metrics.get('roe', 0)
        max_score += 25
        if roe > 0.25:
            score += 25
        elif roe > 0.15:
            score += 20
        elif roe > 0.10:
            score += 15
        elif roe > 0.05:
            score += 10
        elif roe > 0:
            score += 5

        # Net Profit Margin
        npm = metrics.get('profit_margin', 0)
        max_score += 25
        if npm > 0.20:
            score += 25
        elif npm > 0.15:
            score += 20
        elif npm > 0.10:
            score += 15
        elif npm > 0.05:
            score += 10
        elif npm > 0:
            score += 5

        # Operating Margin
        opm = metrics.get('operating_margin', 0)
        max_score += 25
        if opm > 0.25:
            score += 25
        elif opm > 0.18:
            score += 20
        elif opm > 0.12:
            score += 15
        elif opm > 0.05:
            score += 10
        elif opm > 0:
            score += 5

        # ROA
        roa = metrics.get('roa', 0)
        max_score += 25
        if roa > 0.15:
            score += 25
        elif roa > 0.10:
            score += 20
        elif roa > 0.05:
            score += 15
        elif roa > 0.02:
            score += 10
        elif roa > 0:
            score += 5

        return (score / max_score * 100) if max_score > 0 else 0


# ============================================================
# FINANCIAL HEALTH ANALYSIS
# ============================================================

class FinancialHealthAnalyzer:
    """Analyze financial strength and solvency"""

    @staticmethod
    def piotroski_f_score(metrics: Dict) -> Tuple[int, List[str]]:
        """
        Piotroski F-Score (0-9)
        
        Higher score = Stronger fundamentals
        Score >= 7 = Strong
        Score 4-6 = Average
        Score <= 3 = Weak
        
        Very popular screening tool in Indian market
        """
        score = 0
        details = []

        # 1. Positive ROA
        if metrics.get('roa', 0) > 0:
            score += 1
            details.append("âœ… Positive ROA")
        else:
            details.append("âŒ Negative ROA")

        # 2. Positive Operating Cash Flow
        if metrics.get('operating_cash_flow', 0) > 0:
            score += 1
            details.append("âœ… Positive Operating Cash Flow")
        else:
            details.append("âŒ Negative Operating Cash Flow")

        # 3. ROA improving
        if metrics.get('roa', 0) > metrics.get('prev_roa', 0):
            score += 1
            details.append("âœ… ROA improving")
        else:
            details.append("âŒ ROA declining")

        # 4. Cash flow > Net Income (accrual quality)
        if metrics.get('operating_cash_flow', 0) > metrics.get('net_income', 0):
            score += 1
            details.append("âœ… OCF > Net Income (quality earnings)")
        else:
            details.append("âŒ OCF < Net Income")

        # 5. Leverage decreasing
        if metrics.get('debt_to_equity', float('inf')) < metrics.get('prev_debt_to_equity', float('inf')):
            score += 1
            details.append("âœ… Leverage decreasing")
        else:
            details.append("âŒ Leverage increasing")

        # 6. Current ratio improving
        if metrics.get('current_ratio', 0) > metrics.get('prev_current_ratio', 0):
            score += 1
            details.append("âœ… Liquidity improving")
        else:
            details.append("âŒ Liquidity declining")

        # 7. No dilution (shares not increased)
        if metrics.get('shares_outstanding', 0) <= metrics.get('prev_shares_outstanding', float('inf')):
            score += 1
            details.append("âœ… No share dilution")
        else:
            details.append("âŒ Share dilution")

        # 8. Gross margin improving
        if metrics.get('gross_margin', 0) > metrics.get('prev_gross_margin', 0):
            score += 1
            details.append("âœ… Gross margin improving")
        else:
            details.append("âŒ Gross margin declining")

        # 9. Asset turnover improving
        if metrics.get('asset_turnover', 0) > metrics.get('prev_asset_turnover', 0):
            score += 1
            details.append("âœ… Asset turnover improving")
        else:
            details.append("âŒ Asset turnover declining")

        return score, details

    @staticmethod
    def altman_z_score(
        working_capital: float,
        retained_earnings: float,
        ebit: float,
        market_cap: float,
        total_liabilities: float,
        revenue: float,
        total_assets: float
    ) -> Tuple[float, str]:
        """
        Altman Z-Score - Bankruptcy prediction
        
        Z = 1.2A + 1.4B + 3.3C + 0.6D + 1.0E
        
        Z > 2.99 = Safe zone
        1.81 < Z < 2.99 = Grey zone
        Z < 1.81 = Distress zone
        """
        if total_assets <= 0:
            return 0, "insufficient_data"

        A = working_capital / total_assets
        B = retained_earnings / total_assets
        C = ebit / total_assets
        D = market_cap / (total_liabilities if total_liabilities > 0 else 1)
        E = revenue / total_assets

        z = 1.2 * A + 1.4 * B + 3.3 * C + 0.6 * D + 1.0 * E

        if z > 2.99:
            zone = "safe"
        elif z > 1.81:
            zone = "grey"
        else:
            zone = "distress"

        return z, zone

    @staticmethod
    def financial_health_score(metrics: Dict) -> float:
        """Calculate financial health score (0-100)"""
        score = 0
        max_score = 0

        # Debt to Equity
        de = metrics.get('debt_to_equity', 999)
        max_score += 25
        if de < 0.3:
            score += 25
        elif de < 0.5:
            score += 20
        elif de < 1.0:
            score += 15
        elif de < 2.0:
            score += 10
        elif de < 3.0:
            score += 5

        # Current Ratio
        cr = metrics.get('current_ratio', 0)
        max_score += 25
        if cr > 2.0:
            score += 25
        elif cr > 1.5:
            score += 20
        elif cr > 1.0:
            score += 15
        elif cr > 0.7:
            score += 5

        # Interest Coverage
        ic = metrics.get('interest_coverage', 0)
        max_score += 25
        if ic > 10:
            score += 25
        elif ic > 5:
            score += 20
        elif ic > 3:
            score += 15
        elif ic > 1.5:
            score += 10

        # Free Cash Flow Positive
        fcf = metrics.get('free_cash_flow', 0)
        max_score += 25
        if fcf > 0:
            score += 25
        else:
            score += 0

        return (score / max_score * 100) if max_score > 0 else 0


# ============================================================
# GROWTH ANALYSIS
# ============================================================

class GrowthAnalyzer:
    """Analyze company growth metrics"""

    @staticmethod
    def calculate_cagr(
        start_value: float,
        end_value: float,
        years: int
    ) -> float:
        """Compound Annual Growth Rate"""
        if start_value <= 0 or end_value <= 0 or years <= 0:
            return 0
        return (end_value / start_value) ** (1 / years) - 1

    @staticmethod
    def growth_score(metrics: Dict) -> float:
        """Calculate growth score (0-100)"""
        score = 0
        max_score = 0

        # Revenue Growth
        rg = metrics.get('revenue_growth', 0)
        max_score += 30
        if rg > 0.25:
            score += 30
        elif rg > 0.15:
            score += 25
        elif rg > 0.10:
            score += 20
        elif rg > 0.05:
            score += 15
        elif rg > 0:
            score += 10

        # Earnings Growth
        eg = metrics.get('earnings_growth', 0)
        max_score += 30
        if eg > 0.25:
            score += 30
        elif eg > 0.15:
            score += 25
        elif eg > 0.10:
            score += 20
        elif eg > 0.05:
            score += 15
        elif eg > 0:
            score += 10

        # EPS Growth
        epsg = metrics.get('eps_growth', 0)
        max_score += 20
        if epsg > 0.20:
            score += 20
        elif epsg > 0.10:
            score += 15
        elif epsg > 0:
            score += 10

        # Book Value Growth
        bvg = metrics.get('book_value_growth', 0)
        max_score += 20
        if bvg > 0.15:
            score += 20
        elif bvg > 0.10:
            score += 15
        elif bvg > 0:
            score += 10

        return (score / max_score * 100) if max_score > 0 else 0

    @staticmethod
    def growth_consistency(
        values: List[float],
        years: int = 5
    ) -> Dict:
        """
        Analyze growth consistency
        
        Consistent growers command premium in Indian market
        """
        if len(values) < 2:
            return {}

        growth_rates = []
        for i in range(1, len(values)):
            if values[i - 1] > 0:
                gr = (values[i] - values[i - 1]) / values[i - 1]
                growth_rates.append(gr)

        if not growth_rates:
            return {}

        return {
            'avg_growth': np.mean(growth_rates),
            'median_growth': np.median(growth_rates),
            'growth_std': np.std(growth_rates),
            'min_growth': min(growth_rates),
            'max_growth': max(growth_rates),
            'positive_years': sum(1 for g in growth_rates if g > 0),
            'total_years': len(growth_rates),
            'consistency_ratio': sum(1 for g in growth_rates if g > 0) / len(growth_rates),
            'is_consistent': all(g > 0 for g in growth_rates),
        }


# ============================================================
# DIVIDEND ANALYSIS
# ============================================================

class DividendAnalyzer:
    """Analyze dividend characteristics"""

    @staticmethod
    def analyze_dividend(metrics: Dict) -> Dict:
        """
        Comprehensive dividend analysis
        
        Indian market considerations:
        - Dividend income up to â‚¹10 lakh is exempt from tax
        - DDT was abolished; dividends taxed at slab rate
        """
        div_yield = metrics.get('dividend_yield', 0)
        payout = metrics.get('payout_ratio', 0)
        div_growth = metrics.get('dividend_growth', 0)

        # Sustainability check
        is_sustainable = payout < 0.7 and div_yield > 0
        
        # Classification
        if div_yield > 0.05:
            classification = "high_yield"
        elif div_yield > 0.02:
            classification = "moderate_yield"
        elif div_yield > 0:
            classification = "low_yield"
        else:
            classification = "no_dividend"

        # Dividend safety score
        safety_score = 0
        if payout < 0.4:
            safety_score += 40
        elif payout < 0.6:
            safety_score += 30
        elif payout < 0.8:
            safety_score += 15

        if div_growth > 0:
            safety_score += 30

        if metrics.get('free_cash_flow', 0) > 0:
            safety_score += 30

        return {
            'yield': div_yield,
            'payout_ratio': payout,
            'growth_rate': div_growth,
            'classification': classification,
            'is_sustainable': is_sustainable,
            'safety_score': min(safety_score, 100),
        }


# ============================================================
# COMPOSITE FUNDAMENTAL ANALYZER
# ============================================================

class FundamentalAnalyzer:
    """
    Main fundamental analysis class
    
    Combines all analysis modules for comprehensive
    fundamental analysis of Indian stocks
    
    Usage:
        analyzer = FundamentalAnalyzer()
        
        quality = analyzer.get_quality_score(metrics)
        valuation = analyzer.get_valuation(metrics)
        report = analyzer.generate_report(symbol, metrics)
    """

    def __init__(self):
        self.valuation = ValuationAnalyzer()
        self.profitability = ProfitabilityAnalyzer()
        self.health = FinancialHealthAnalyzer()
        self.growth = GrowthAnalyzer()
        self.dividend = DividendAnalyzer()

    def get_quality_score(self, metrics: Dict) -> QualityScore:
        """
        Calculate comprehensive quality score
        
        Combines profitability, growth, health, valuation, efficiency
        """
        # Individual scores
        prof_score = self.profitability.profitability_score(metrics)
        growth_score = self.growth.growth_score(metrics)
        health_score = self.health.financial_health_score(metrics)

        # Valuation score
        pe = metrics.get('pe_ratio', 0)
        if 0 < pe < 15:
            val_score = 90
        elif pe < 25:
            val_score = 70
        elif pe < 40:
            val_score = 50
        elif pe < 60:
            val_score = 30
        else:
            val_score = 10

        # Efficiency (simplified)
        efficiency_score = min(
            metrics.get('asset_turnover', 0) / 2 * 100,
            100
        )

        # Piotroski
        piotroski, piotroski_details = self.health.piotroski_f_score(metrics)

        # Altman Z-Score
        z_score, z_zone = self.health.altman_z_score(
            metrics.get('working_capital', 0),
            metrics.get('retained_earnings', 0),
            metrics.get('ebit', 0),
            metrics.get('market_cap', 0),
            metrics.get('total_liabilities', 0),
            metrics.get('revenue', 0),
            metrics.get('total_assets', 1),
        )

        # Weighted composite score
        total_score = (
            prof_score * 0.25 +
            growth_score * 0.25 +
            health_score * 0.20 +
            val_score * 0.20 +
            efficiency_score * 0.10
        )

        # Determine rating
        if total_score >= 80 and piotroski >= 7:
            rating = FundamentalRating.STRONG_BUY
        elif total_score >= 65:
            rating = FundamentalRating.BUY
        elif total_score >= 45:
            rating = FundamentalRating.HOLD
        elif total_score >= 30:
            rating = FundamentalRating.SELL
        else:
            rating = FundamentalRating.STRONG_SELL

        # Strengths & Weaknesses
        strengths = []
        weaknesses = []

        if prof_score > 70:
            strengths.append(f"Strong profitability ({prof_score:.0f}/100)")
        elif prof_score < 30:
            weaknesses.append(f"Weak profitability ({prof_score:.0f}/100)")

        if growth_score > 70:
            strengths.append(f"Strong growth ({growth_score:.0f}/100)")
        elif growth_score < 30:
            weaknesses.append(f"Weak growth ({growth_score:.0f}/100)")

        if health_score > 70:
            strengths.append(f"Strong financial health ({health_score:.0f}/100)")
        elif health_score < 30:
            weaknesses.append(f"Poor financial health ({health_score:.0f}/100)")

        if pe > 0 and pe < 20:
            strengths.append(f"Reasonable valuation (PE: {pe:.1f})")
        elif pe > 50:
            weaknesses.append(f"Expensive valuation (PE: {pe:.1f})")

        if piotroski >= 7:
            strengths.append(f"High Piotroski score ({piotroski}/9)")
        elif piotroski <= 3:
            weaknesses.append(f"Low Piotroski score ({piotroski}/9)")

        if z_zone == "safe":
            strengths.append(f"Safe Altman Z-Score ({z_score:.2f})")
        elif z_zone == "distress":
            weaknesses.append(f"Distress Altman Z-Score ({z_score:.2f})")

        return QualityScore(
            symbol=metrics.get('symbol', ''),
            total_score=total_score,
            profitability_score=prof_score,
            growth_score=growth_score,
            financial_health_score=health_score,
            valuation_score=val_score,
            efficiency_score=efficiency_score,
            piotroski_score=piotroski,
            altman_z_score=z_score,
            rating=rating,
            strengths=strengths,
            weaknesses=weaknesses,
        )

    def get_valuation(self, metrics: Dict) -> ValuationResult:
        """Get comprehensive valuation"""
        current_price = metrics.get('current_price', 0)
        eps = metrics.get('eps', 0)
        book_value = metrics.get('book_value', 0)

        # Graham Number
        graham = self.valuation.graham_number(eps, book_value)

        # DCF
        fcf_list = metrics.get('free_cash_flows', [])
        dcf_result = {}
        intrinsic_value = graham  # Default to Graham

        if fcf_list:
            dcf_result = self.valuation.dcf_valuation(
                fcf_list,
                growth_rate=metrics.get('revenue_growth', 0.10),
                shares_outstanding=metrics.get('shares_outstanding', 1),
            )
            if dcf_result.get('intrinsic_value_per_share', 0) > 0:
                intrinsic_value = (graham + dcf_result['intrinsic_value_per_share']) / 2

        mos = self.valuation.margin_of_safety(current_price, intrinsic_value)

        if mos > 30:
            val_rating = "significantly_undervalued"
        elif mos > 10:
            val_rating = "undervalued"
        elif mos > -10:
            val_rating = "fairly_valued"
        elif mos > -30:
            val_rating = "overvalued"
        else:
            val_rating = "significantly_overvalued"

        return ValuationResult(
            symbol=metrics.get('symbol', ''),
            current_price=current_price,
            intrinsic_value=intrinsic_value,
            graham_number=graham,
            margin_of_safety=mos,
            pe_ratio=metrics.get('pe_ratio', 0),
            pb_ratio=metrics.get('pb_ratio', 0),
            peg_ratio=self.valuation.peg_ratio(
                metrics.get('pe_ratio', 0),
                metrics.get('earnings_growth', 0)
            ),
            ev_ebitda=metrics.get('ev_to_ebitda', 0),
            ps_ratio=metrics.get('ps_ratio', 0),
            valuation_rating=val_rating,
            details=dcf_result,
        )

    def generate_report(self, symbol: str, metrics: Dict) -> Dict:
        """Generate comprehensive fundamental analysis report"""
        quality = self.get_quality_score(metrics)
        valuation = self.get_valuation(metrics)

        dupont = self.profitability.dupont_analysis(
            metrics.get('net_income', 0),
            metrics.get('revenue', 0),
            metrics.get('total_assets', 1),
            metrics.get('total_equity', 1),
        )

        dividend_analysis = self.dividend.analyze_dividend(metrics)

        return {
            'symbol': symbol,
            'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M'),
            'quality_score': quality,
            'valuation': valuation,
            'dupont': dupont,
            'dividend': dividend_analysis,
            'rating': quality.rating.value,
            'total_score': quality.total_score,
            'strengths': quality.strengths,
            'weaknesses': quality.weaknesses,
        }

    def screen_by_fundamentals(
        self,
        stocks_data: List[Dict],
        min_score: float = 60
    ) -> List[Dict]:
        """Screen stocks by quality score"""
        results = []

        for metrics in stocks_data:
            try:
                quality = self.get_quality_score(metrics)
                if quality.total_score >= min_score:
                    results.append({
                        'symbol': metrics.get('symbol', ''),
                        'total_score': quality.total_score,
                        'rating': quality.rating.value,
                        'piotroski': quality.piotroski_score,
                        'profitability': quality.profitability_score,
                        'growth': quality.growth_score,
                        'health': quality.financial_health_score,
                        'valuation': quality.valuation_score,
                    })
            except Exception as e:
                logger.error(f"Error screening {metrics.get('symbol', '')}: {e}")

        results.sort(key=lambda x: x['total_score'], reverse=True)
        return results

