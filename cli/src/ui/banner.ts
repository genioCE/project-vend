import figlet from "figlet";
import chalk from "chalk";
import { c } from "./colors.js";

const GREY_STEPS = [32, 56, 88, 128, 176, 224, 255];
const TAGLINE_STEPS = [56, 128, 224];
const FRAME_MS = 50;

function grey(level: number) {
  return chalk.rgb(level, level, level);
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export async function printBanner(): Promise<void> {
  const art = figlet.textSync("CORPUS", { font: "ANSI Shadow" });
  const artLines = art.split("\n").filter((l) => l.trim().length > 0);
  const version = "v0.2.0";
  const tagline = "private intelligence for your writing";
  const padding = "  ";

  const rowCount = artLines.length;
  const totalArtFrames = rowCount - 1 + GREY_STEPS.length;
  const totalFrames = totalArtFrames + TAGLINE_STEPS.length;
  const displayLines = rowCount + 1; // art rows + tagline row

  // Non-TTY: static print, no animation
  if (!process.stdout.isTTY) {
    for (const line of artLines) {
      console.log(`${padding}${chalk.white(line)}`);
    }
    console.log(`${padding}${c.dim(tagline)}  ${c.dim(version)}`);
    console.log();
    return;
  }

  process.stdout.write("\x1b[?25l"); // hide cursor
  process.stdout.write("\n");

  // Reserve vertical space
  for (let i = 0; i < displayLines; i++) {
    process.stdout.write("\n");
  }

  for (let frame = 0; frame < totalFrames; frame++) {
    // Move cursor to top of art block
    process.stdout.write(`\x1b[${displayLines}A`);

    for (let row = 0; row < rowCount; row++) {
      process.stdout.write("\r\x1b[2K");
      const stepIndex = frame - row;
      if (stepIndex >= 0 && stepIndex < GREY_STEPS.length) {
        process.stdout.write(
          `${padding}${grey(GREY_STEPS[stepIndex])(artLines[row])}`,
        );
      } else if (stepIndex >= GREY_STEPS.length) {
        process.stdout.write(`${padding}${chalk.white(artLines[row])}`);
      }
      process.stdout.write("\n");
    }

    // Tagline row
    process.stdout.write("\r\x1b[2K");
    const tagFrame = frame - totalArtFrames;
    if (tagFrame >= 0 && tagFrame < TAGLINE_STEPS.length) {
      const g = grey(TAGLINE_STEPS[tagFrame]);
      process.stdout.write(`${padding}${g(tagline)}  ${g(version)}`);
    } else if (tagFrame >= TAGLINE_STEPS.length) {
      process.stdout.write(
        `${padding}${c.dim(tagline)}  ${c.dim(version)}`,
      );
    }
    process.stdout.write("\n");

    await sleep(FRAME_MS);
  }

  process.stdout.write("\x1b[?25h"); // show cursor
  console.log();
}
