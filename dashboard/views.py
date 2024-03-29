import csv
import json
import random
import requests
from django.http import HttpResponse, JsonResponse
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.conf import settings
from .models import Portfolio, StockHolding
from riskprofile.models import RiskProfile
from riskprofile.views import risk_profile
import yfinance as yf
import datetime
# AlphaVantage API
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.fundamentaldata import FundamentalData
import subprocess as sp
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from typing import List
from itertools import combinations

def get_alphavantage_key():
  alphavantage_keys = [
    settings.ALPHAVANTAGE_KEY1,
    settings.ALPHAVANTAGE_KEY2,
    settings.ALPHAVANTAGE_KEY3,

  ]
  return random.choice(alphavantage_keys)



@login_required
def dashboard(request):
    if RiskProfile.objects.filter(user=request.user).exists():
        try:
            risk_profile = RiskProfile.objects.get(user=request.user)
            portfolio = Portfolio.objects.get(user=request.user)
        except:
            portfolio = Portfolio.objects.create(user=request.user)

        portfolio.update_investment()
        holding_companies = StockHolding.objects.filter(portfolio=portfolio)
        holdings = []
        sectors = [[], []]
        sector_wise_investment = {}
        stocks = [[], []]
        intraday_signals = {}  # Store intraday buy and sell signals for each stock
        
        for c in holding_companies:
            company_symbol = c.company_symbol
            company_name = c.company_name
            number_shares = c.number_of_shares
            investment_amount = c.investment_amount
            average_cost = investment_amount / number_shares
            holdings.append({
                'CompanySymbol': company_symbol,
                'CompanyName': company_name,
                'NumberShares': number_shares,
                'InvestmentAmount': investment_amount,
                'AverageCost': average_cost,
            })

            # Fetch intraday stock data using yfinance

            stocks[0].append(round((investment_amount / portfolio.total_investment) * 100, 2))
            stocks[1].append(company_symbol)
            if c.sector in sector_wise_investment:
                sector_wise_investment[c.sector] += investment_amount
            else:
                sector_wise_investment[c.sector] = investment_amount
        for sec in sector_wise_investment.keys():
            sectors[0].append(round((sector_wise_investment[sec] / portfolio.total_investment) * 100, 2))
            sectors[1].append(sec)
        risk_category = risk_profile.category
        tickers =['AAPL','NVDA','BTC','GOOGL','MSFT','AMZN']
        years_simulated = 2
        set_size = 3
        risk_tolerance = risk_category
        data =  fetch_data(tickers, years_simulated)
        columns_to_use = data.columns.tolist()
        columns_to_use.remove('Date')
        if 'risk_free_rate' in columns_to_use:
            columns_to_use.remove('risk_free_rate')

        results_df = portfolio_metrics(data, columns_to_use, set_size=set_size, years_simulated=years_simulated)

        # Apply filters based on risk tolerance
        if risk_tolerance == "Aggressive" or "Assertive":
            # Prioritize portfolios with higher Total_Return and Sharpe_Ratio
            results_df = (
                results_df.sort_values(by='Sharpe_Ratio',ascending=False)
                        .sort_values(by='Sortino_Ratio', ascending=False)
                        .sort_values(by='Total_Return', ascending=False)
            )
        elif risk_tolerance == "Conservative":
            # Prioritize portfolios with lower VaR, Max_Drawdown, and higher Total_Return
            results_df = (
                results_df.sort_values(by='VaR', ascending=True)
                        .sort_values(by='Total_Return', ascending=False)
                        
            )

        sorted_results = results_df.head(1)
        sorted_results_dict = sorted_results.to_dict(orient='records')
        print(sorted_results_dict)
        # Adding
        print(risk_tolerance)
        news = fetch_news()
        ###
        context = {
            'holdings': holdings,
            'totalInvestment': portfolio.total_investment,
            'stocks': stocks,
            'sectors': sectors,
            'news': news,
            'recommendation':sorted_results_dict
  # Pass intraday buy and sell signals to the template
        } 

        return render(request, 'dashboard/dashboard.html', context)
    else:
        return redirect(risk_profile)

def get_portfolio_insights(request):
    try:
        portfolio = Portfolio.objects.get(user=request.user)
        holding_companies = StockHolding.objects.filter(portfolio=portfolio)
        print(portfolio)
        print(holding_companies)
        portfolio_beta = 0
        portfolio_pe = 0
        
        for c in holding_companies:
            stock_data = yf.Ticker(c.company_symbol)
            
            # Get beta and PE ratio from yfinance
            try:
                beta = "{:.2f}".format(stock_data.info['beta'])
            except KeyError:
                beta = 0  # Set a default value if beta is not available
            try:
                pe_ratio = "{:.2f}".format(stock_data.info['trailingPE'])
            except KeyError:
                pe_ratio = 0  # Set a default value if PE ratio is not available
            
            # Update portfolio beta and PE ratio
            portfolio_beta += float(beta) * (c.investment_amount / portfolio.total_investment)
            portfolio_pe += float(pe_ratio) * (c.investment_amount / portfolio.total_investment)
        
        return JsonResponse({"PortfolioBeta": portfolio_beta, "PortfolioPE": portfolio_pe})
    except Exception as e:
        return JsonResponse({"Error": str(e)})

def update_values(request):
    try:
        portfolio = Portfolio.objects.get(user=request.user)
        current_value = 0
        unrealized_pnl = 0
        growth = 0
        holding_companies = StockHolding.objects.filter(portfolio=portfolio)
        stockdata = {}
        for c in holding_companies:
            # Fetching stock data using yfinance
            stock_data = yf.Ticker(c.company_symbol)
            last_trading_price = stock_data.history(period="1d")['Close'].iloc[-1]  # Get last trading price
            pnl = (last_trading_price * c.number_of_shares) - c.investment_amount
            net_change = pnl / c.investment_amount
            stockdata[c.company_symbol] = {
                'LastTradingPrice': last_trading_price,
                'PNL': pnl,
                'NetChange': net_change * 100
            }
            current_value += (last_trading_price * c.number_of_shares)
            unrealized_pnl += pnl
        if portfolio.total_investment != 0:
            growth = unrealized_pnl / portfolio.total_investment
        return JsonResponse({
            "StockData": stockdata, 
            "CurrentValue": current_value,
            "UnrealizedPNL": unrealized_pnl,
            "Growth": growth * 100
        })
    except Exception as e:
        return JsonResponse({"Error": str(e)})


def get_financials(request):
    try:
        symbol = request.GET.get('symbol')
        stock_data = yf.Ticker(symbol)
        data = yf.download(symbol, period="1y")

        #Get the 52-week high from the 'High' column
        data_inf = yf.download(symbol, period="1y").info
        fifty_two_week_low = data['High'].min()
        fifty_two_week_high = data['High'].max()

        # Format Beta if it's not None
        beta = stock_data.info.get('beta')
        formatted_beta = "{:.2f}".format(beta) if beta is not None else "N/A"

        financials = {
            "52WeekHigh": fifty_two_week_high,
            "52WeekLow": fifty_two_week_low,
            "Beta": formatted_beta,
            "BookValue": stock_data.info.get('bookValue'),
            "EBITDA": stock_data.info.get('ebitda'),
            "EVToEBITDA": stock_data.info.get('enterpriseToEbitda'),
            "OperatingMarginTTM": stock_data.info.get('operatingMargins'),
            "PERatio": stock_data.info.get('forwardPE'),
            "PriceToBookRatio": stock_data.info.get('priceToBook'),
            "ProfitMargin": stock_data.info.get('profitMargins'),
            "ReturnOnAssetsTTM": stock_data.info.get('returnOnAssets'),
            "ReturnOnEquityTTM": stock_data.info.get('returnOnEquity'),
            "Sector": "N/A"  # Yahoo Finance doesn't provide sector information directly
        }
        
        return JsonResponse({ "financials": financials })
    except Exception as e:
        return JsonResponse({"Error": str(e)})


def add_holding(request):
    if request.method == "POST":
        try:
            portfolio = Portfolio.objects.get(user=request.user)
            holding_companies = StockHolding.objects.filter(portfolio=portfolio)
            company_symbol = request.POST['company'].split('(')[1].split(')')[0]
            company_name = request.POST['company'].split('(')[0].strip()
            number_stocks = int(request.POST['number-stocks'])

            # Fetching stock data using yfinance
            stock_data = yf.Ticker(company_symbol)
            history = stock_data.history(period="1d")  # Fetch daily historical data

            if history.empty:
                raise ValueError("Error: No data available for the specified symbol")

            # Extracting the latest closing price
            buy_price = history['Close'].iloc[-1]

            found = False
            for c in holding_companies:
                if c.company_symbol == company_symbol:
                    c.buying_value.append([buy_price, number_stocks])
                    c.save()
                    found = True

            if not found:
                c = StockHolding.objects.create(
                    portfolio=portfolio, 
                    company_name=company_name, 
                    company_symbol=company_symbol,
                    number_of_shares=number_stocks
                )
                c.buying_value.append([buy_price, number_stocks])
                c.save()

            return HttpResponse("Success")
        except ValueError as e:
            print("Value Error:", e)
            return HttpResponse("Value Error: " + str(e))
        except Exception as e:
            print("Error:", e)
            return HttpResponse("Error: " + str(e))

def send_company_list(request):
  with open('nasdaq-listed.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    rows = []
    for row in csv_reader:
      if line_count == 0:
        line_count += 1
      else:
        rows.append([row[0], row[1]])
        line_count += 1
  return JsonResponse({"data": rows})

def fetch_news():
  query_params = {
    "country": "us",
    "category": "business",
    "sortBy": "top",
    "apiKey": settings.NEWSAPI_KEY
  }
  main_url = "https://newsapi.org/v2/top-headlines"
  # fetching data in json format
  res = requests.get(main_url, params=query_params)
  open_bbc_page = res.json()
  # getting all articles in a string article
  article = open_bbc_page["articles"]
  results = []
  for ar in article:
    results.append([ar["title"], ar["description"], ar["url"]])
  # Make news as 2 at a time to show on dashboard
  news = zip(results[::2], results[1::2])
  if len(results) % 2:
    news.append((results[-1], None))
  return news


def backtesting(request):
  print('Function Called')
  try:
    output = sp.check_output("quantdom", shell=True)
  except sp.CalledProcessError:
    output = 'No such command'
  return HttpResponse("Success")


def fetch_data(tickers: List[str], years_simulated: int):
    current_date = datetime.datetime.today().strftime('%Y-%m-%d')
    start_date = datetime.datetime.today() - datetime.timedelta(days=years_simulated * 365)
    start_date = start_date.strftime('%Y-%m-%d')
    end_date = current_date

    tickers_str = ' '.join(tickers)
    data = yf.download(tickers=tickers_str, start=start_date, end=end_date, progress=False)['Adj Close']
    data.reset_index(inplace=True)

    risk_free_rate = yf.Ticker("^TNX").history(period="5y")['Close']
    risk_free_rate = risk_free_rate.rename('risk_free_rate')
    if isinstance(risk_free_rate.index, pd.DatetimeIndex) and risk_free_rate.index.tzinfo is not None:
        risk_free_rate.index = risk_free_rate.index.tz_localize(None)

    data = pd.merge(data, risk_free_rate, left_on='Date', right_index=True, how='left')
    data['risk_free_rate'] = data['risk_free_rate'].ffill()
    data = data.dropna(axis=1, how='all')

    return data

def portfolio_metrics(dataframe, columns_to_use, years_simulated=5, set_size=3, var_confidence=0.05, cvar_confidence=0.05, trading_days_per_year=252, user=None):
    # Initialize results DataFrame
    results_list = []

    # Check if SPY exists for Portfolio Beta calculation
    include_beta = 'SPY' in dataframe.columns

    # Generate all unique combinations
    all_combinations = list(combinations(columns_to_use, set_size))

    # Filter DataFrame based on years simulated
    latest_date = dataframe['Date'].max()
    earliest_date = pd.Timestamp(latest_date) - pd.DateOffset(years=years_simulated)
    filtered_df = dataframe[dataframe['Date'] >= earliest_date]

    for selected_columns in all_combinations:
        returns = filtered_df[list(selected_columns)].pct_change(fill_method=None).dropna()
        risk_free_rate = filtered_df['risk_free_rate'].loc[returns.index].mean() / 100  # Convert to decimal
        weights = np.array([1./set_size] * set_size)

        # Adjust weights based on user's risk tolerance and investment goals
        if user:
            if user.risk_tolerance == 'Aggressive':
                weights *= 1.2
            elif user.risk_tolerance == 'Conservative':
                weights *= 0.8
            
            if user.investment_goals == 'Short-Term Gains':
                # Emphasize total return for short-term gains
                metrics_score = total_return
            elif user.investment_goals == 'Long-Term Growth':
                # Emphasize CAGR for long-term growth
                metrics_score = cagr

        # Sharpe Ratio
        expected_return = np.sum(returns.mean() * weights) * trading_days_per_year
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * trading_days_per_year, weights)))
        sharpe_ratio = (expected_return - risk_free_rate) / portfolio_volatility

        # VaR and CVaR
        portfolio_return_series = (returns * weights).sum(axis=1)
        var_value = portfolio_return_series.quantile(var_confidence)
        cvar_value = portfolio_return_series[portfolio_return_series <= var_value].mean()

        # Maximum Drawdown
        cumulative_returns = (1 + portfolio_return_series).cumprod()
        max_value = cumulative_returns.cummax()
        drawdowns = cumulative_returns / max_value - 1
        max_drawdown = drawdowns.min()

        # Sortino Ratio
        negative_returns = portfolio_return_series[portfolio_return_series < 0]
        downside_deviation = negative_returns.std() * np.sqrt(trading_days_per_year)
        sortino_ratio = (expected_return - risk_free_rate) / downside_deviation

        # Portfolio Beta
        if include_beta:
            spy_returns = filtered_df['SPY'].diff() / filtered_df['SPY'].shift(1)
            aligned_data = pd.concat([portfolio_return_series, spy_returns], axis=1).dropna()
            portfolio_beta = aligned_data.iloc[:, 0].cov(aligned_data.iloc[:, 1]) / aligned_data.iloc[:, 1].var()

        # Total Return and CAGR
        total_return = cumulative_returns.iloc[-1] - 1  # subtract 1 to convert to percentage
        cagr = (cumulative_returns.iloc[-1] ** (1 / years_simulated)) - 1  # again, subtract 1 for percentage

        # Append to results
        metrics = {f'stock_{i+1}': stock for i, stock in enumerate(selected_columns)}
        metrics['Sharpe_Ratio'] = sharpe_ratio
        metrics['VaR'] = var_value
        metrics['CVaR'] = cvar_value
        metrics['Max_Drawdown'] = max_drawdown
        metrics['Sortino_Ratio'] = sortino_ratio
        metrics['Total_Return'] = total_return
        metrics['CAGR'] = cagr
        if include_beta:
            metrics['Portfolio_Beta'] = portfolio_beta

        results_list.append(metrics)

    # Convert results to DataFrame
    results_df = pd.DataFrame(results_list)

    return results_df
