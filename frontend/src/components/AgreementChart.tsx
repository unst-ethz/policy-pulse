import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { score } from './ui';

const colors = ['#2373b5', '#9c5ca4', '#248878', '#c2763d', '#6c70a4', '#677c37'];
export function AgreementChart({
  data,
  countries,
  names,
}: {
  data: Array<Record<string, number | string | boolean | null>>;
  countries: string[];
  names: Record<string, string>;
}) {
  return (
    <div className="chart">
      <ResponsiveContainer width="100%" height={360}>
        <LineChart data={data} margin={{ top: 20, right: 24, bottom: 20, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e6ebf1" />
          <XAxis
            dataKey="year"
            type="number"
            domain={['dataMin', 'dataMax']}
            allowDecimals={false}
            tick={{ fontSize: 12 }}
          />
          <YAxis domain={[0, 1]} tick={{ fontSize: 12 }} width={45} />
          <Tooltip
            formatter={(value, name) => [score(typeof value === 'number' ? value : null), name]}
            labelFormatter={(year, payload) =>
              payload?.[0]?.payload?.session
                ? `Session ${payload[0].payload.session} · ${year}`
                : String(year)
            }
          />
          <Legend />
          {countries.map((country, i) => (
            <Line
              key={country}
              type="linear"
              dataKey={country}
              name={names[country] ?? country}
              stroke={colors[i % colors.length]}
              strokeWidth={2}
              connectNulls
              isAnimationActive={false}
              dot={(props) => {
                const { cx, cy, payload, value } = props;
                return value == null ? (
                  <g key={`${country}-${payload.year}-${payload.session}`} />
                ) : (
                  <circle
                    key={`${country}-${payload.year}-${payload.session}`}
                    cx={cx}
                    cy={cy}
                    r={3}
                    fill={payload.special ? '#141e2b' : colors[i % colors.length]}
                  />
                );
              }}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
