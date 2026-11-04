import type { ReactNode } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { Count } from "./data";

export const colors = [
  "#35d4a4",
  "#66a9f5",
  "#f3bd69",
  "#f17c69",
  "#b6a0fa",
  "#7ed0df",
];
const tooltip = {
  background: "#15212b",
  border: "1px solid #415360",
  borderRadius: 8,
  color: "#fff",
};
const axis = { fill: "#9aadb8", fontSize: 11 };
export function ChartCard({
  title,
  empty,
  children,
}: {
  title: string;
  empty?: boolean;
  children: ReactNode;
}) {
  return (
    <section className="panel metric-chart">
      <h2>{title}</h2>
      {empty ? (
        <p className="muted metric-empty">За выбранный период нет данных</p>
      ) : (
        children
      )}
    </section>
  );
}
export function Bars({
  data,
  horizontal = false,
}: {
  data: Count[];
  horizontal?: boolean;
}) {
  const height = horizontal ? Math.max(220, data.length * 35 + 40) : 250;
  return (
    <div className={horizontal ? "metric-scroll-chart" : undefined}>
      <ResponsiveContainer width="100%" height={height} minWidth={0}>
        <BarChart
          data={data}
          layout={horizontal ? "vertical" : "horizontal"}
          margin={{ top: 12, right: 16, bottom: 4, left: 0 }}
        >
          <CartesianGrid stroke="#2d3d47" strokeDasharray="3 4" />
          <XAxis
            type={horizontal ? "number" : "category"}
            dataKey={horizontal ? undefined : "name"}
            tick={axis}
            allowDecimals={false}
          />
          <YAxis
            type={horizontal ? "category" : "number"}
            dataKey={horizontal ? "name" : undefined}
            tick={axis}
            width={horizontal ? 145 : 36}
            tickFormatter={(name) =>
              String(name).length > 21 ? String(name).slice(0, 20) + "…" : name
            }
            allowDecimals={false}
          />
          <Tooltip
            contentStyle={tooltip}
            itemStyle={{ color: "#fff" }}
            cursor={{ fill: "#30404b", opacity: 0.5 }}
          />
          <Bar
            dataKey="count"
            name="Инциденты"
            fill={colors[0]}
            radius={horizontal ? [0, 4, 4, 0] : [4, 4, 0, 0]}
            isAnimationActive={false}
          />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
export function Donut({ data }: { data: Count[] }) {
  return (
    <ResponsiveContainer width="100%" height={250} minWidth={0}>
      <PieChart>
        <Pie
          data={data}
          dataKey="count"
          nameKey="name"
          innerRadius="49%"
          outerRadius="76%"
          isAnimationActive={false}
        >
          {data.map((item, i) => (
            <Cell
              key={item.name}
              fill={colors[i % colors.length]}
              stroke="none"
            />
          ))}
        </Pie>
        <Tooltip contentStyle={tooltip} itemStyle={{ color: "#fff" }} />
        <Legend wrapperStyle={{ fontSize: 12 }} />
      </PieChart>
    </ResponsiveContainer>
  );
}
export function Series({
  data,
  series,
  percent = false,
}: {
  data: object[];
  series: { key: string; name: string }[];
  percent?: boolean;
}) {
  return (
    <ResponsiveContainer width="100%" height={260} minWidth={0}>
      <LineChart
        data={data}
        margin={{ top: 12, right: 14, bottom: 3, left: 0 }}
      >
        <CartesianGrid stroke="#2d3d47" strokeDasharray="3 4" />
        <XAxis
          dataKey="date"
          tick={axis}
          minTickGap={20}
          tickFormatter={(date) => date.slice(5).split("-").reverse().join(".")}
        />
        <YAxis
          tick={axis}
          width={43}
          domain={percent ? [0, 100] : [0, "auto"]}
          unit={percent ? "%" : undefined}
          allowDecimals={false}
        />
        <Tooltip contentStyle={tooltip} />
        {!percent && <Legend wrapperStyle={{ fontSize: 12 }} />}
        {series.map(({ key, name }, i) => (
          <Line
            key={key}
            dataKey={key}
            name={name}
            stroke={colors[i % colors.length]}
            dot={{ r: 2 }}
            strokeWidth={2}
            isAnimationActive={false}
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}
