import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { ProbabilityBars } from "./ProbabilityBars";

describe("ProbabilityBars", () => {
  it("lists classes by descending probability and honours top", () => {
    render(<ProbabilityBars probabilities={{ none: 0.1, Center: 0.7, Loc: 0.2 }} top={2} />);
    const items = within(screen.getByLabelText("class probabilities")).getAllByRole("listitem");
    expect(items).toHaveLength(2);
    expect(items[0]).toHaveTextContent("Center70.0%");
    expect(items[1]).toHaveTextContent("Loc20.0%");
  });
});
